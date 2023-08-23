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

        self.logger.info(f"Starting model executor using '{self.device}' for tensor processing.")
        self.model_repository = InMemoryModelRepository(self.node, self.device)

    async def save_model(self, model_meta_data):
        await self.model_repository.save_model(model_meta_data)
        self.logger.info(f"Saved model '{model_meta_data.model_name}'.")
