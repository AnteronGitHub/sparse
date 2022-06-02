import torch
from torch import nn

from sparse.config_manager import MasterConfigManager, WorkerConfigManager
from sparse.roles.master import Master
from sparse.roles.worker import Worker
from sparse.dl.gradient_calculator import GradientCalculator
from sparse.dl.serialization import encode_offload_request, decode_offload_response
from sparse.dl.utils import get_device

from models.index import FIRST_SPLIT

class SplitTrainingClient(Master, Worker):
    def __init__(self, model, loss_fn, optimizer, train_dataloader, classes, task_executor):
        worker_config_manager = WorkerConfigManager()
        config_manager = MasterConfigManager()
        config_manager.listen_address = worker_config_manager.listen_address
        config_manager.listen_port = worker_config_manager.listen_port

        Master.__init__(self, config_manager = config_manager)
        Worker.__init__(self, task_executor = task_executor, config_manager = config_manager)

        self.device = get_device()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.classes = classes

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
    model_kind = "basic"
    model = FIRST_SPLIT[model_kind]()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    first_split_calculator = GradientCalculator(model=model,
                                                loss_fn=nn.CrossEntropyLoss(),
                                                optimizer=torch.optim.SGD(model.parameters(), lr=1e-3))
    if model_kind == "vgg":
        from datasets.cifar10 import load_CIFAR10_dataset
        (
            train_dataloader,
            test_dataloader,
            classes,
        ) = load_CIFAR10_dataset()
    else:
        from datasets.mnist_fashion import load_mnist_fashion_dataset
        (
            train_dataloader,
            test_dataloader,
            classes,
        ) = load_mnist_fashion_dataset()

    SplitTrainingClient(model=model,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        train_dataloader=train_dataloader,
                        classes=classes,
                        task_executor=first_split_calculator).train()

    # TODO: evaluate
