import torch
from torch import nn

from sparse.node.master import Master
from sparse.node.worker import Worker
from sparse.dl.gradient_calculator import GradientCalculator

from models.index import FIRST_SPLIT

class SplitTrainingClient(Master, Worker):
    def __init__(self, task_executor):
        Master.__init__(self)
        Worker.__init__(self, task_executor = task_executor)

if __name__ == "__main__":
    model_kind = "vgg"
    model = FIRST_SPLIT[model_kind]()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    first_split_calculator = GradientCalculator(model=model,
                                                loss_fn=nn.CrossEntropyLoss(),
                                                optimizer=torch.optim.SGD(model.parameters(), lr=1e-3))

    SplitTrainingClient(task_executor=first_split_calculator).start()

    # TODO: evaluate
