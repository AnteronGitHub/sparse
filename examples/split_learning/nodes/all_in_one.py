import asyncio

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class AllInOne():
    def __init__(self, dataset, classes, model, loss_fn, optimizer):
        self.dataset = dataset
        self.classes = classes
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def train(self, batches, batch_size, epochs, log_file_prefix):
        print(f"Using {self.device} for processing")

        self.model.to(self.device)

        progress_bar = tqdm(total=epochs*batches*batch_size,
                            unit='samples',
                            unit_scale=True)
        for t in range(epochs):
            for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)

                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                progress_bar.update(len(X))
                if batch + 1 >= batches:
                    break

        progress_bar.close()
