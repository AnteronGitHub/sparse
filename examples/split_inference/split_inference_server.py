import torch
from torch import nn
from torch.autograd import Variable

from sparse.node.worker import Worker
from sparse.task_executor import TaskExecutor

from serialization import decode_offload_request, encode_offload_response

from models import NeuralNetwork_local, NeuralNetwork_server
from utils import get_device


class InferenceCalculator(TaskExecutor):
    def __init__(self, config_path_server, weight_server, compressionProps):
        super().__init__()
        self.device = get_device()
        client_model = NeuralNetwork_local("config/yolov3_local.cfg", compressionProps)
        self.model = NeuralNetwork_server(config_path_server, compressionProps, client_model.prevfiltersGet())
        self.model.load_state_dict(torch.load(weight_server))

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




if __name__ == "__main__":
    config_path_server = "config/yolov3_server.cfg"
    weight_server = "weights/yolov3_server.paths"
    compressionProps = {} ###
    compressionProps['feature_compression_factor'] = 4 ### resolution compression factor, compress by how many times
    compressionProps['resolution_compression_factor'] = 1 ###layer compression factor, reduce by how many times TBD

    split_training_server = Worker(task_executor=InferenceCalculator(config_path_server, weight_server, compressionProps))
    split_training_server.start()
