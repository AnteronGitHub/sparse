import torch
from torch import nn

from sparse.roles.master import Master
from sparse.roles.worker import Worker
from sparse.dl.gradient_calculator import GradientCalculator
from sparse.dl.serialization import encode_offload_request, decode_offload_response
from sparse.dl.utils import get_device

from models.index import FIRST_SPLIT

class SplitTrainingClient(Master, Worker):
    def __init__(self, task_executor):
        Master.__init__(self)
        Worker.__init__(self, task_executor = task_executor)

if __name__ == "__main__":
    model_kind = "basic"
    model = FIRST_SPLIT[model_kind]()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    first_split_calculator = GradientCalculator(model=model,
                                                loss_fn=nn.CrossEntropyLoss(),
                                                optimizer=torch.optim.SGD(model.parameters(), lr=1e-3))

    SplitTrainingClient(task_executor=first_split_calculator).start()

    # TODO: evaluate
