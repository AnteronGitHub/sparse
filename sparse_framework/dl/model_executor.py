import torch

from sparse_framework.task_executor import TaskExecutor

from .models.model_loader import ModelLoader
from .utils import get_device

class ModelExecutor(TaskExecutor):
    def __init__(self, model_name : str, partition : str):
        super().__init__()
        self.device = get_device()
        self.model_name = model_name
        self.partition = partition

        self.model = None

    def start(self):
        """Initialize executor by transferring the model to the processor memory."""
        super().start()

        model_loader = ModelLoader(self.node.config_manager.model_server_address,
                                   self.node.config_manager.model_server_port)

        self.model, self.loss_fn, self.optimizer = model_loader.load_model(self.model_name,
                                                                           self.partition)
        self.logger.info(f"Downloaded model '{self.model_name}' partition '{self.partition}'")

        num_parameters = 0
        for param in self.model.parameters():
            num_parameters += param.nelement()
        self.logger.info(f"Model executor using model '{type(self.model).__name__}' with {num_parameters} parameters using {self.device} for processing")
        self.model.to(self.device)

