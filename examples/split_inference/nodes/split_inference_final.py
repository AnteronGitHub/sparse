import torch
from torch import nn
from torch.autograd import Variable

from sparse.node.worker import Worker
from sparse.task_executor import TaskExecutor

from serialization import decode_offload_request, encode_offload_response

from models.yolov3 import YOLOv3_server
from utils import get_device

class InferenceCalculator(TaskExecutor):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = get_device()

    def start(self):
        """Initialize executor by transferring the model to the processor memory."""
        super().start()
        self.logger.info(f"Task executor using {self.device} for processing")
        self.model.to(self.device)

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single forward computation for the offloaded layers."""
        split_layer = decode_offload_request(input_data).to(self.device)
        pred = self.model(split_layer)

        # Result serialization
        return encode_offload_response(pred.to("cpu").detach())

class SplitInferenceFinal(Worker):
    def __init__(self):
        compressionProps = {}
        compressionProps['feature_compression_factor'] = 4
        compressionProps['resolution_compression_factor'] = 1

        Worker.__init__(self,
                        task_executor = InferenceCalculator(YOLOv3_server(compressionProps)))
