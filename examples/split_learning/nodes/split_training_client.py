import asyncio
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from sparse_framework.node.master import Master
from sparse_framework.dl.gradient_calculator import GradientCalculator
from sparse_framework.dl.utils import get_device
from sparse_framework.dl.serialization import encode_offload_request, decode_offload_response, encode_offload_request_pruned
from sparse_framework.stats.monitor_client import MonitorClient

import numpy as np

class SplitTrainingClient(Master):
    def __init__(self, dataset, model, loss_fn, optimizer, benchmark = True):
        Master.__init__(self)
        self.dataset = dataset
        # self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = get_device()
        self.model = model.to(self.device)
        if benchmark:
            self.monitor_client = MonitorClient()
        else:
            self.monitor_client = None

    async def start(self, batch_size, batches, depruneProps, log_file_prefix):
        if self.monitor_client is not None:
            self.monitor_client.start_benchmark(log_file_prefix)

        total_epochs = 0
        for entry in depruneProps:
            total_epochs += depruneProps[entry]['epochs']

        progress_bar = tqdm(total=batch_size*batches*total_epochs,
                            unit='samples',
                            unit_scale=True)

        for entry in depruneProps:
            epochs = depruneProps[entry]['epochs']
            pruneState = depruneProps[entry]['pruneState']
            budget = depruneProps[entry]['budget']
            for t in range(epochs):
                for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                    X, y = X.to(self.device), y.to(self.device)

                    # Local forward pass
                    model_return = self.model(X, local=True)

                    pred = model_return[0].to("cpu").detach()  # partial model output
                    ############################
                    # quantization/compression TBD
                    ############################
                    prune_filter = model_return[1].to(
                        "cpu").detach()  # the prune filter in training

                    # Offloaded layers
                    upload_data, filter_to_send = self.compress_with_pruneFilter(
                        pred, prune_filter, budget)

                    input_data = encode_offload_request_pruned(
                        upload_data, y.to("cpu"), filter_to_send, budget)

                    result_data = await self.task_deployer.deploy_task(input_data)
                    split_grad, loss = decode_offload_response(result_data)

                    progress_bar.update(len(X))
                    if self.monitor_client is not None:
                        self.monitor_client.batch_processed(len(X))

                    if batch + 1 >= batches:
                        break

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
