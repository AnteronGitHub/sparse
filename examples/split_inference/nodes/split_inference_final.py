from sparse.node.worker import Worker
from sparse.dl.inference_calculator import InferenceCalculator

class SplitInferenceFinal(Worker):
    def __init__(self, model):
        Worker.__init__(self,
                        task_executor = InferenceCalculator(model))
