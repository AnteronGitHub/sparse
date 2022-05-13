import torch
from torch import nn

from sparse.roles.master import Master

from models.index import FIRST_SPLIT
from serialization import encode_offload_request, decode_offload_response
from utils import get_device


class SplitTrainingClient(Master):
    def __init__(self, model_kind: str = "basic"):
        super().__init__()
        self.device = get_device()
        self.model = FIRST_SPLIT[model_kind]()
        if model_kind == "vgg":
            from datasets.cifar10 import load_CIFAR10_dataset
            (
                self.train_dataloader,
                self.test_dataloader,
                self.classes,
            ) = load_CIFAR10_dataset()
        else:
            from datasets.mnist_fashion import load_mnist_fashion_dataset
            (
                self.train_dataloader,
                self.test_dataloader,
                self.classes,
            ) = load_mnist_fashion_dataset()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    def train(self, epochs: int = 5):
        self.logger.info(
            f"Starting training using {self.device} for local computations"
        )

        # Transfer model to device
        model = self.model.to(self.device)
        model.train()

        for t in range(epochs):
            self.logger.info(f"--------- Epoch {t+1:>2d} ----------")
            size = len(self.train_dataloader.dataset)

            for batch, (X, y) in enumerate(self.train_dataloader):
                # Transfer training data to device memory
                X = X.to(self.device)

                # Local forward propagation
                split_vals = model(X)

                # Offloaded layers
                input_data = encode_offload_request(split_vals.to("cpu").detach(), y)
                result_data = self.task_deployer.deploy_task(input_data)
                split_grad, loss = decode_offload_response(result_data)

                # Local back propagation
                split_grad = split_grad.to(self.device)
                self.optimizer.zero_grad()
                split_vals.backward(split_grad)
                self.optimizer.step()

                if batch % 100 == 0:
                    current = batch * len(X)
                    self.logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        self.logger.info("Done!")


if __name__ == "__main__":
    SplitTrainingClient("vgg").train()

    # TODO: evaluate
