import asyncio
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from sparse_framework.node.master import Master
from sparse_framework.dl.utils import get_device
from sparse_framework.dl.serialization import encode_offload_request, decode_offload_response, encode_offload_request_pruned
from sparse_framework.dl import ModelLoader

import numpy as np

from utils import parse_arguments, get_depruneProps, get_deprune_epochs, _get_benchmark_log_file_prefix
from datasets import DatasetRepository

class LearningClient(Master):
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

    def load_model(self):
        model_loader = ModelLoader(self.config_manager.model_server_address,
                                   self.config_manager.model_server_port)

        self.model, self.loss_fn, self.optimizer = model_loader.load_model(self.model_name,
                                                                           self.partition,
                                                                           self.compressionProps,
                                                                           self.use_compression)
        self.model = self.model.to(self.device)
        self.logger.info(f"Downloaded model '{self.model_name}' partition '{self.partition}' with compression props '{self.compressionProps}' and using compression '{self.use_compression}'")

    async def start(self, batch_size, batches, depruneProps, log_file_prefix, use_compression, epochs, verbose = False):
        total_epochs = get_deprune_epochs(depruneProps)

        if verbose:
            progress_bar = tqdm(total=batch_size*batches*total_epochs,
                                unit='samples',
                                unit_scale=True)
        else:
            progress_bar = None

        if use_compression:
            phases = depruneProps
        else:
            phases = [{'epochs': epochs, 'budget': None}]

        for phase, prop in enumerate(phases):
            epochs = prop['epochs']
            budget = prop['budget']
            for t in range(epochs):
                offset = 0 if phase == 0 and t == 0 else 1 # Ensures that an extra batch is processed in the first epoch since one batch is for warming up
                for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                    X = X.to(self.device)

                    if use_compression:
                        # Masked prediction (quantization TBD)
                        pred, prune_filter = self.model(X)
                        payload, mask = self.compress_with_pruneFilter(pred.to(self.device).detach(),
                                                                       prune_filter.to(self.device).detach(),
                                                                       budget)
                        input_data = encode_offload_request_pruned(payload, y, mask, budget)
                    else:
                        pred = self.model(X)
                        input_data = encode_offload_request(pred, y)


                    # Offloaded layers
                    while True:
                        result_data = await self.task_deployer.deploy_task(input_data)
                        if result_data is None:
                            self.logger.error(f"Broken pipe error. Re-executing...")
                            if self.monitor_client is not None:
                                self.monitor_client.broken_pipe_error()
                        else:
                            break

                    split_grad, loss = decode_offload_response(result_data)
                    if use_compression:
                        split_grad = self.decompress_with_pruneFilter(split_grad, mask, budget)

                    split_grad = split_grad.to(self.device)

                    # Back Propagation
                    self.optimizer.zero_grad()
                    pred.backward(split_grad)
                    self.optimizer.step()

                    # Logging
                    if progress_bar is not None:
                        progress_bar.update(len(X))
                    else:
                        self.logger.info(f"Processed batch of {len(X)} samples")

                    # Benchmarks
                    if self.monitor_client is not None:
                        if self.warmed_up:
                            self.monitor_client.batch_processed(len(X), loss)
                        else:
                            self.warmed_up = True
                            self.monitor_client.start_benchmark(log_file_prefix)

                    if batch + offset >= batches:
                        break

        if progress_bar is not None:
            progress_bar.close()
        if self.monitor_client is not None:
            self.monitor_client.stop_benchmark()
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)

    def compress_with_pruneFilter(self, pred, prune_filter, budget):

        mask = torch.square(torch.sigmoid(prune_filter.squeeze()))
        topk = torch.topk(mask, budget)
        compressedPred = torch.index_select(pred, 1, topk.indices.sort().values)
        return compressedPred, mask

    def decompress_with_pruneFilter(self, pred, mask, budget):

        a = torch.mul(mask.repeat([128,1]).t(), torch.eye(128).to(self.device))
        b = a.index_select(1, mask.topk(budget).indices.sort().values)
        b = torch.where(b>0.0, 1.0, 0.0).to(self.device)
        decompressed_pred = torch.einsum('ij,bjlm->bilm', b, pred)

        return decompressed_pred

if __name__ == '__main__':
    args = parse_arguments()

    compressionProps = {}
    ### resolution compression factor, compress by how many times
    compressionProps['feature_compression_factor'] = args.feature_compression_factor
    ###layer compression factor, reduce by how many times TBD
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor
    depruneProps = get_depruneProps(args)
    partition = "client"
    use_compression = bool(args.use_compression)

    dataset, classes = DatasetRepository().get_dataset(args.dataset)
    log_file_prefix = _get_benchmark_log_file_prefix(args, "datasource", get_deprune_epochs(depruneProps))

    learning_client = LearningClient(dataset=dataset,
                                     model_name=args.model_name,
                                     partition=partition,
                                     compressionProps=compressionProps,
                                     use_compression=use_compression)
    learning_client.load_model()
    asyncio.run(learning_client.start(args.batch_size,
                                      args.batches,
                                      depruneProps,
                                      log_file_prefix=log_file_prefix,
                                      use_compression=use_compression,
                                      epochs=int(args.epochs)))
