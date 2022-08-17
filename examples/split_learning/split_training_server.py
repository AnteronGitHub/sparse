import torch
from torch import nn

from sparse.node.worker import Worker
from sparse.dl.gradient_calculator import GradientCalculator

from models.vgg import VGG_server

if __name__ == "__main__":
    model = VGG_server()
    second_split_calculator = GradientCalculator(model=model,
                                                 loss_fn=nn.CrossEntropyLoss(),
                                                 optimizer=torch.optim.SGD(model.parameters(), lr=1e-3))
    split_training_server = Worker(task_executor=second_split_calculator)
    split_training_server.start()
