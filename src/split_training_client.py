import torch
from torch import nn

from datasets.mnist_fashion import load_mnist_fashion_dataset
from models.neural_network import NeuralNetwork_local
from serialization import encode_offload_request, decode_offload_response
from roles.master import Master, TaskDeployer
from utils import get_device, use_legacy_asyncio

class SplitTrainingClient(Master):
    def __init__(self, upstream_host='127.0.0.1'):
        super().__init__(upstream_host=upstream_host, legacy_asyncio = use_legacy_asyncio())
        self.device = get_device()
        self.model = NeuralNetwork_local()
        self.train_dataloader, self.test_dataloader, self.classes = load_mnist_fashion_dataset()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    def train(self, epochs : int = 5):
        print("Starting training")

        # Transfer model to device
        model = self.model.to(self.device)
        model.train()

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            size = len(self.train_dataloader.dataset)

            for batch, (X, y) in enumerate(self.train_dataloader):
                # Transfer training data to device memory
                X = X.to(self.device)

                # Local forward propagation
                split_vals = model(X)

                # Offloaded layers
                input_data = encode_offload_request(split_vals.to('cpu').detach(), y)
                result_data = self.task_deployer.deploy_task(input_data)
                split_grad, loss = decode_offload_response(result_data)

                # Local back propagation
                split_grad = split_grad.to(self.device)
                self.optimizer.zero_grad()
                split_vals.backward(split_grad)
                self.optimizer.step()

                if batch % 100 == 0:
                    current = batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        print("Done!")


if __name__ == '__main__':
    SplitTrainingClient(upstream_host='127.0.0.1').train()

    # TODO: evaluate
