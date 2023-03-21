from sparse_framework.node.master import Master
from sparse_framework.node.worker import Worker
from sparse_framework.dl.gradient_calculator import GradientCalculator

class SplitTrainingIntermediate(Master, Worker):
    def __init__(self, model, loss_fn, optimizer, depruneProps):
        task_executor = GradientCalculator(model=model,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           depruneProps = depruneProps)
        Master.__init__(self)
        Worker.__init__(self, task_executor = task_executor)

