import torch
from torch import nn
from tqdm import tqdm

from sparse.node.master import Master
from sparse.dl.serialization import encode_offload_request, decode_offload_response

import asyncio

class SplitTrainingDataSource(Master):
    def __init__(self, train_dataloader, classes):
        Master.__init__(self)

        self.train_dataloader = train_dataloader
        self.classes = classes

    async def train(self, epochs: int = 1):
        self.logger.info(f"Starting streaming input data for training")

        print(f"Training {epochs} epochs with {len(train_dataloader.dataset)} samples")
        for t in range(epochs):
            self.logger.info(f"--------- Epoch {t+1:>2d} ----------")
            size = len(self.train_dataloader.dataset)

            for batch, (X, y) in enumerate(self.train_dataloader):
                input_data = encode_offload_request(X, y)
                result_data = await self.task_deployer.deploy_task(input_data)
                split_grad, loss = decode_offload_response(result_data)

                if batch % 100 == 0:
                    current = batch * len(X)
                    self.logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        self.logger.info("Done!")

if __name__ == "__main__":
    model_kind = "vgg"
    if model_kind == "vgg":
        from datasets.cifar10 import load_CIFAR10_dataset
        train_dataloader, classes = load_CIFAR10_dataset()
    else:
        from datasets.mnist_fashion import load_mnist_fashion_dataset
        (
            train_dataloader,
            test_dataloader,
            classes,
        ) = load_mnist_fashion_dataset()
    data_source = SplitTrainingDataSource(train_dataloader=train_dataloader,
                                          classes=classes)
    asyncio.run(data_source.train())

    # TODO: evaluate
