import torch
from torch import nn
from torch.autograd import Variable

from sparse.roles.worker import Worker
from sparse.roles.task_executor import TaskExecutor

from serialization import decode_offload_request, encode_offload_response

from models import NeuralNetwork_server
from utils import get_device


class InferenceCalculator():
    def __init__(self, model_kind: str = "basic"):
        super().__init__()
        self.device = get_device()
        self.model = NeuralNetwork_server

    def start(self):
        """Initialize executor by transferring the model to the processor memory."""
        super().start()
        self.logger.info(f"Task executor using {self.device} for processing")
        self.model.to(self.device)

    def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single forward computation for the offloaded layers."""
        split_layer = decode_offload_request(input_data)
        pred = self.model(split_layer)

        # Result serialization
        return encode_offload_response(pred.grad.to("cpu").detach())




if __name__ == "__main__":
    split_training_server = Worker(task_executor=InferenceCalculator("vgg"))
    split_training_server.start()