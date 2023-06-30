import asyncio
from contextlib import nullcontext
import time
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from sparse_framework.node.master import Master
from sparse_framework.stats.monitor_client import MonitorClient
from sparse_framework.dl import DatasetRepository, ModelLoader
from sparse_framework.dl.utils import get_device

from benchmark import parse_arguments, get_depruneProps, _get_benchmark_log_file_prefix
from serialization import decode_offload_response, \
                          decode_offload_inference_response, \
                          encode_offload_request, \
                          encode_offload_request_pruned, \
                          encode_offload_inference_request, \
                          encode_offload_inference_request_pruned

from compression_utils import compress_with_pruneFilter, decompress_with_pruneFilter
from models.compression_utils_vgg import EncodingUnit

class DepruneClient(Master):
    def __init__(self,
                 application,
                 dataset,
                 model_name : str,
                 partition : str,
                 compressionProps : dict,
                 benchmark : bool = True):
        Master.__init__(self, benchmark=benchmark)

        self.device = get_device()

        self.application = application
        self.dataset = dataset
        self.model_name = model_name
        self.partition = partition
        self.encoder = EncodingUnit(compressionProps, in_channel=128)
        self.warmed_up = False

        self.model = None

    def load_model(self):
        model_loader = ModelLoader(self.config_manager.model_server_address,
                                   self.config_manager.model_server_port)

        self.model, self.loss_fn, self.optimizer = model_loader.load_model(self.model_name,
                                                                           self.partition)
        self.logger.info(f"Downloaded model '{self.model_name}' partition '{self.partition}'")

        self.model = self.model.to(self.device)
        self.encoder = self.encoder.to(self.device)

        if self.is_learning():
            self.logger.info(f"Training the model.")
            self.model.train()
        else:
            self.logger.info(f"Inferring with the model.")

    def is_learning(self):
        return self.application == 'learning'

    async def start(self, batch_size, batches, depruneProps, log_file_prefix, verbose = False):
        if self.monitor_client is not None:
            self.monitor_client.start_benchmark(log_file_prefix)

        if verbose:
            inferences_to_be_run = batch_size * batches
            progress_bar = tqdm(total=inferences_to_be_run,
                                unit='samples',
                                unit_scale=True)
        else:
            progress_bar = None

        with nullcontext() if self.is_learning() else torch.no_grad():
            for phase, prop in enumerate(depruneProps):
                epochs = prop['epochs']
                pruneState = prop['pruneState']
                budget = prop['budget']
                for t in range(epochs):
                    for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                        X = X.to(self.device)

                        pred = self.model(X)

                        if pruneState:
                            pred, prune_filter = self.encoder(pred)
                            payload, mask = compress_with_pruneFilter(pred.to("cpu").detach(),
                                                                      prune_filter.to("cpu").detach(),
                                                                      budget)

                            if self.is_learning():
                                input_data = encode_offload_request_pruned(payload, y, mask, budget)
                            else:
                                input_data = encode_offload_inference_request_pruned(payload, mask, budget)
                        else:
                            if self.is_learning():
                                input_data = encode_offload_request(pred.to("cpu").detach(), y)
                            else:
                                input_data = encode_offload_inference_request(pred.to("cpu").detach())

                        # Offloaded layers
                        while True:
                            result_data = await self.task_deployer.deploy_task(input_data)
                            if result_data is None:
                                self.logger.error(f"Broken pipe error. Re-executing...")
                                if self.monitor_client is not None:
                                    self.monitor_client.broken_pipe_error()
                            else:
                                break

                        if self.is_learning():
                            split_grad, loss = decode_offload_response(result_data)
                            split_grad = decompress_with_pruneFilter(split_grad, mask, budget, self.device)

                            # Back Propagation
                            split_grad = split_grad.to(self.device)

                            self.optimizer.zero_grad()
                            pred.backward(split_grad)
                            self.optimizer.step()
                        else:
                            final_prediction = decode_offload_inference_response(result_data)
                            loss = None

                        if self.monitor_client is not None:
                            self.monitor_client.batch_processed(len(X))
                        if progress_bar is not None:
                            progress_bar.update(len(X))
                        else:
                            self.logger.info(f"Processed batch of {len(X)} samples. Loss: {loss}")

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

    dataset, classes = DatasetRepository().get_dataset(args.dataset)

    deprune_client = DepruneClient(application=args.application,
                                   dataset=dataset,
                                   model_name=args.model_name,
                                   partition=partition,
                                   compressionProps=compressionProps)
    deprune_client.load_model()
    asyncio.run(deprune_client.start(args.batch_size,
                                     args.batches,
                                     depruneProps,
                                     log_file_prefix=_get_benchmark_log_file_prefix(args, "datasource")))
