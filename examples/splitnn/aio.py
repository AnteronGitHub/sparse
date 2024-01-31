import asyncio
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from sparse_framework import Node
from sparse_framework.dl import get_device, DatasetRepository, ModelLoader

from utils import parse_arguments, _get_benchmark_log_file_prefix

class SplitNNAllInOne(Node):
    def __init__(self, dataset, model_name, partition, application, benchmark = True):
        Node.__init__(self, benchmark=benchmark)
        self.device = get_device()

        self.dataset = dataset
        self.model_name = model_name
        self.partition = partition
        self.application = application
        self.warmed_up = False

    def load_model(self):
        model_loader = ModelLoader(self.config_manager.model_server_address,
                                   self.config_manager.model_server_port)

        self.model, self.loss_fn, self.optimizer = model_loader.load_model(self.model_name,
                                                                           self.partition)
        self.model = self.model.to(self.device)
        self.logger.info(f"Using {self.device} for processing")
        self.logger.info(f"Downloaded model '{self.model_name}' partition '{self.partition}'.")

    async def process_batch(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        pred = self.model(X)

        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    async def start(self, batch_size, batches, epochs, log_file_prefix, verbose = False):
        if verbose:
            progress_bar = tqdm(total=epochs*batches*batch_size,
                                unit='samples',
                                unit_scale=True)
        else:
            progress_bar = None

        for t in range(epochs):
            offset = 0 if t == 0 else 1 # Ensures that an extra batch is processed in the first epoch since one batch is for warming up
            for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                loss = await asyncio.create_task(self.process_batch(X, y))

                if progress_bar is not None:
                    progress_bar.update(len(X))
                else:
                    self.logger.info(f"Processed batch of {len(X)} samples")
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

if __name__ == '__main__':
    args = parse_arguments()

    partition = "aio"
    dataset, classes = DatasetRepository().get_dataset(args.dataset)

    splitnn_aio = SplitNNAllInOne(application=args.application,
                                  dataset=dataset,
                                  model_name=args.model_name,
                                  partition=partition)
    splitnn_aio.load_model()
    asyncio.run(splitnn_aio.start(args.batch_size,
                                  args.batches,
                                  args.epochs,
                                  _get_benchmark_log_file_prefix(args, "aio")))
