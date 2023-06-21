import asyncio
import time

import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from sparse_framework.node.master import Master
from sparse_framework.stats.monitor_client import MonitorClient
from sparse_framework.dl import ModelLoader

from datasets.yolov3 import YOLOv3Dataset
from models.yolov3 import YOLOv3_local
from utils import get_device, ImageLoading, non_max_suppression, save_detection
from torch.utils.data import DataLoader
from sparse_framework.dl.serialization import encode_offload_inference_request, encode_offload_inference_request_pruned

from benchmark import parse_arguments, get_depruneProps, _get_benchmark_log_file_prefix

class InferenceClient(Master):
    def __init__(self,
                 dataset,
                 model_name : str,
                 partition : str,
                 compressionProps : dict,
                 use_compression : bool,
                 benchmark : bool = True):
        Master.__init__(self, benchmark=benchmark)

        self.device = get_device()

        self.dataset = dataset
        self.model_name = model_name
        self.partition = partition
        self.compressionProps = compressionProps
        self.use_compression = use_compression
        self.warmed_up = False

        self.model = None

    def load_model(self):
        model_loader = ModelLoader(self.config_manager.model_server_address,
                                   self.config_manager.model_server_port)

        self.model, self.loss_fn, self.optimizer = model_loader.load_model(self.model_name,
                                                                           self.partition,
                                                                           self.compressionProps,
                                                                           self.use_compression)
        self.model = self.model.to(self.device)
        self.logger.info(f"Downloaded model '{self.model_name}' partition '{self.partition}' with compression props '{self.compressionProps}' and using compression '{self.use_compression}'")

    def compress_with_pruneFilter(self, pred, prune_filter, budget):
        mask = torch.square(torch.sigmoid(prune_filter.squeeze()))
        topk = torch.topk(mask, budget)
        compressedPred = torch.index_select(pred, 1, topk.indices.sort().values)

        return compressedPred, mask

    async def infer(self, batch_size, batches, depruneProps, epochs, log_file_prefix = None, verbose = False):
        if self.monitor_client is not None:
            self.monitor_client.start_benchmark(log_file_prefix)

        if verbose:
            inferences_to_be_run = batch_size * batches
            progress_bar = tqdm(total=inferences_to_be_run,
                                unit='inferences',
                                unit_scale=True)
        else:
            progress_bar = None

        if self.use_compression:
            phases = depruneProps
        else:
            phases = [{'epochs': epochs, 'budget': None, 'pruneState': 0}]

        with torch.no_grad():
            for phase, prop in enumerate(phases):
                epochs = prop['epochs']
                pruneState = prop['pruneState']
                budget = prop['budget']
                for t in range(epochs):
                    for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                        X, y = X.to(self.device), y.to(self.device)

                        if pruneState:
                            pred, prune_filter = self.model(X)
                            pred, mask = self.compress_with_pruneFilter(pred.to("cpu").detach(),
                                                                        prune_filter.to("cpu").detach(),
                                                                        budget)
                            input_data = encode_offload_inference_request_pruned(pred, mask, budget)
                        else:
                            pred = self.model(X)
                            input_data = encode_offload_inference_request(pred.to("cpu").detach())

                        result_data = await self.task_deployer.deploy_task(input_data)

                        if self.monitor_client is not None:
                            self.monitor_client.batch_processed(len(X))
                        if progress_bar is not None:
                            progress_bar.update(len(X))
                        else:
                            self.logger.info(f"Processed batch of {len(X)} samples")

                        if batch + 1 >= batches:
                            break

        if progress_bar is not None:
            progress_bar.close()
        if self.monitor_client is not None:
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)

if __name__ == "__main__":
    args = parse_arguments()

    compressionProps = {}
    compressionProps['feature_compression_factor'] = args.feature_compression_factor
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor

    depruneProps = get_depruneProps(args)
    partition = "client"
    use_compression = bool(args.use_compression)

    from datasets import DatasetRepository
    dataset, classes = DatasetRepository().get_dataset(args.dataset)

    inference_client = InferenceClient(dataset=dataset,
                                       model_name=args.model_name,
                                       partition=partition,
                                       compressionProps=compressionProps,
                                       use_compression=use_compression)
    inference_client.load_model()
    asyncio.run(inference_client.infer(args.batch_size,
                                       args.batches,
                                       depruneProps,
                                       int(args.epochs),
                                       log_file_prefix=_get_benchmark_log_file_prefix(args)))
