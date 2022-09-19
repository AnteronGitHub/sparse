import torch
from torch import nn

from sparse.node.master import Master
from sparse.node.worker import Worker
from sparse.dl.gradient_calculator import GradientCalculator

class SplitTrainingIntermediate(Master, Worker):
    def __init__(self, model, loss_fn, optimizer):
        task_executor = GradientCalculator(model=model,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        Master.__init__(self)
        Worker.__init__(self, task_executor = task_executor)

