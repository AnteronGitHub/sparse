import asyncio
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from sparse.node.master import Master
from sparse.dl.gradient_calculator import GradientCalculator
from sparse.dl.utils import get_device
from sparse.dl.serialization import encode_offload_request, decode_offload_response
from sparse.stats.monitor_client import MonitorClient

class SplitTrainingClient(Master):
    def __init__(self, dataset, model, loss_fn, optimizer, benchmark = True):
        Master.__init__(self)
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = get_device()
        if benchmark:
            self.monitor_client = MonitorClient()
        else:
            self.monitor_client = None

    async def start(self, batch_size, batches, epochs):
        if self.monitor_client is not None:
            self.monitor_client.start_benchmark()

        progress_bar = tqdm(total=batch_size*batches*epochs,
                            unit='samples',
                            unit_scale=True)
        for t in range(epochs):
            for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                X, y = X.to(self.device), y.to(self.device)

                # Local forward pass
                pred = self.model(X)

                input_data = encode_offload_request(pred.to("cpu").detach(), y.to("cpu"))
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

