from sparse_framework.task_executor import TaskExecutor

from .model_repository import InMemoryModelRepository
from ..utils import get_device

class ModelExecutor(TaskExecutor):
    def __init__(self):
        super().__init__()

        self.device = get_device()

    def start(self):
        super().start()

        self.logger.info(f"Starting model executor using '{self.device}' for tensor processing.")
