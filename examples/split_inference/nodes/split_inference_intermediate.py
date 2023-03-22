from sparse_framework.node.master import Master
from sparse_framework.node.worker import Worker
from sparse_framework.dl.inference_calculator import InferenceCalculator

class SplitInferenceIntermediate(Master, Worker):
    def __init__(self, model, depruneProps):

        Master.__init__(self)
        Worker.__init__(self, task_executor = InferenceCalculator(model, depruneProps))
