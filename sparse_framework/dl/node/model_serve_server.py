from sparse_framework import Worker

from ..executor import TensorExecutor
from ..protocols import ModelServeServerProtocol
from ..serving import InMemoryModelRepository
from ..utils import get_device

class ModelServeServer(Worker):
    def __init__(self):
        rx_protocol_factory = lambda task_queue, stats_queue: \
                                    lambda: ModelServeServerProtocol(self, task_queue, stats_queue)
        super().__init__(rx_protocol_factory, task_executor=TensorExecutor)

        self.model_repository = None

    def get_model_repository(self):
        if (self.model_repository is None):
            self.model_repository = InMemoryModelRepository(self, get_device())
        return self.model_repository

