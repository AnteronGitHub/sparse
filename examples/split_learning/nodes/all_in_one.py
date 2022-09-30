import asyncio
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from sparse.dl.utils import get_device
from sparse.stats.monitor_client import MonitorClient

class AllInOne():
    def __init__(self, dataset, classes, model, loss_fn, optimizer, benchmark = True):
        self.dataset = dataset
        self.classes = classes
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = get_device()
        if benchmark:
            print(f"Benchmarking the suite")
            self.monitor_client = MonitorClient()
        else:
            print(f"Not benchmarking the suite")
            self.monitor_client = None

    async def process_batch(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        pred = self.model(X)

        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    async def train(self, batches, batch_size, epochs, log_file_prefix):
        print(f"Using {self.device} for processing")
        self.model.to(self.device)

        if self.monitor_client is not None:
            await self.monitor_client.start_benchmark(log_file_prefix)

        progress_bar = tqdm(total=epochs*batches*batch_size,
                            unit='samples',
                            unit_scale=True)
        for t in range(epochs):
            for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                await asyncio.create_task(self.process_batch(X, y))

                if self.monitor_client is not None:
                    await self.monitor_client.batch_processed(len(X))

                progress_bar.update(len(X))
                if batch + 1 >= batches:
                    break

        progress_bar.close()
        if self.monitor_client is not None:
            print("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)
