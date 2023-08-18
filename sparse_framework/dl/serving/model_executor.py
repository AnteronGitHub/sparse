from sparse_framework.task_executor import TaskExecutor

from .model_repository import InMemoryModelRepository
from ..utils import get_device

class ModelExecutor(TaskExecutor):
    def __init__(self):
        super().__init__()

        self.device = get_device()
        self.model_repository = None

    def start(self):
        super().start()

        self.model_repository = InMemoryModelRepository(self.node, self.device)

    async def save_model(self):
        await self.model_loader.save_model(self.model.to("cpu"), self.model_name, self.partition)
        self.logger.info(f"Saved model '{self.model_name}' partition '{self.partition}'")
