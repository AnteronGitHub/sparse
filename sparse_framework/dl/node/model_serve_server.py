from sparse_framework import Worker

from ..executor import TensorExecutor
from ..protocols import ModelServeServerProtocol
from ..serving import InMemoryModelRepository
from ..utils import get_device

class ModelServeServer(Worker):
    def __init__(self):
        Worker.__init__(self, task_executor=TensorExecutor, rx_protocol=ModelServeServerProtocol)

    async def start(self):
        self.model_repository = InMemoryModelRepository(self, get_device())
        await super().start()

