import asyncio
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from sparse_framework.node.master import Master
from sparse_framework.dl.utils import get_device
from sparse_framework.dl.serialization import encode_offload_request, decode_offload_response, encode_offload_request_pruned
from sparse_framework.dl.model_loader import ModelLoader
from sparse_framework.stats.monitor_client import MonitorClient

import numpy as np

from utils import parse_arguments, get_depruneProps, get_deprune_epochs, _get_benchmark_log_file_prefix
from datasets import DatasetRepository
from models import ModelTrainingRepository

class LearningClient(Master):
    def __init__(self,
                 dataset,
                 model_name : str,
                 partition : str,
                 compressionProps : dict,
                 use_compression : bool,
                 benchmark : bool = True):
        Master.__init__(self)

        self.device = get_device()

        self.dataset = dataset
        self.model_name = model_name
        self.partition = partition
        self.compressionProps = compressionProps
        self.use_compression = use_compression
        self.warmed_up = False

        if benchmark:
            self.monitor_client = MonitorClient()
        else:
            self.monitor_client = None

    def load_model(self):
        model_loader = ModelLoader(self.config_manager.model_server_address,
                                   self.config_manager.model_server_port)

        self.model, self.loss_fn, self.optimizer = model_loader.load_model(self.model_name,
                                                                           self.partition,
                                                                           self.compressionProps,
                                                                           self.use_compression)
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

        for prop in phases:
            epochs = prop['epochs']
            budget = prop['budget']
            for t in range(epochs):
                for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                    X = X.to(self.device)

                    if use_compression:
                        # Masked prediction (quantization TBD)
                        pred, prune_filter = self.model(X)
                        payload, mask = self.compress_with_pruneFilter(pred.to("cpu").detach(),
                                                                       prune_filter.to("cpu").detach(),
                                                                       budget)
                        input_data = encode_offload_request_pruned(payload, y, mask, budget)
                    else:
                        pred = self.model(X)
                        input_data = encode_offload_request(pred, y)


                    # Offloaded layers
                    result_data = await self.task_deployer.deploy_task(input_data)

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

                    if batch >= batches:
                        break

        if progress_bar is not None:
            progress_bar.close()
        if self.monitor_client is not None:
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)

    def compress_with_pruneFilter(self, pred, prune_filter, budget):

        compressedPred = torch.tensor([])
        mask = torch.square(torch.sigmoid(prune_filter.squeeze())).to('cpu')
        masknp = mask.detach().numpy()
        partitioned = np.partition(masknp, -budget)[-budget]
        for entry in range(len(mask)):
            if mask[entry] >= partitioned:
                predRow = pred[:, entry, :, :].unsqueeze(dim=1)
                compressedPred = torch.cat((compressedPred, predRow), 1)

        return compressedPred, mask

    def decompress_with_pruneFilter(self, pred, mask, budget):

        decompressed_pred = torch.tensor([]).to(self.device)
        a_row = pred[:,0,:,:].unsqueeze(dim=1)
        zeroPad = torch.zeros(a_row.shape).to(self.device)
        masknp = mask.to('cpu').detach().numpy()
        partitioned = np.partition(masknp, -budget)[-budget]
        count = 0
        for entry in range(len(mask)):
            if mask[entry] >= partitioned:
                predRow = pred[:,count,:,:].unsqueeze(dim=1).to(self.device)
                decompressed_pred = torch.cat((decompressed_pred, predRow), 1)
                count += 1
            else:
                decompressed_pred = torch.cat((decompressed_pred, zeroPad), 1)

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
