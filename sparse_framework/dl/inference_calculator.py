
import torch
from torch.autograd import Variable
import numpy as np

from sparse_framework.task_executor import TaskExecutor

from .serialization import decode_offload_inference_request, \
                           encode_offload_inference_request, \
                           decode_offload_inference_response, \
                           encode_offload_inference_response, \
                           encode_offload_inference_request_pruned
from .serialization import decode_offload_inference_request_pruned


from .utils import get_device
from .models.model_loader import ModelLoader

class InferenceCalculator(TaskExecutor):
    def __init__(self, model_name : str, partition : str, compressionProps : dict, use_compression : bool):
        super().__init__()
        self.device = get_device()
        self.model_name = model_name
        self.partition = partition
        self.compressionProps = compressionProps
        self.use_compression = use_compression

        self.model = None

    def start(self):
        """Initialize executor by transferring the model to the processor memory."""
        super().start()

        model_loader = ModelLoader(self.node.config_manager.model_server_address,
                                   self.node.config_manager.model_server_port)

        self.model, self.loss_fn, self.optimizer = model_loader.load_model(self.model_name,
                                                                           self.partition,
                                                                           self.compressionProps,
                                                                           self.use_compression)
        self.logger.info(f"Downloaded model '{self.model_name}' partition '{self.partition}' with compression props '{self.compressionProps}' and using compression '{self.use_compression}'")

        num_parameters = 0
        for param in self.model.parameters():
            num_parameters += param.nelement()
        self.logger.info(f"Inferring with model {type(self.model).__name__} with {num_parameters} parameters using {self.device} for processing")
        self.model.to(self.device)
        self.logger.info(f"Task executor using {self.device} for processing")

        self.model.to(self.device)

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single forward computation for the offloaded layers."""
        split_layer = decode_offload_inference_request(input_data)
        pred = self.model(split_layer.to(self.device))

        if self.task_deployer:
            self.logger.debug("Deploying to the next worker further")

            offload_input_data = encode_offload_inference_request(pred.to("cpu").detach())
            result_data = await self.task_deployer.deploy_task(offload_input_data)
        else:
            self.logger.debug("Not deploying task any further")
            result_data = encode_offload_inference_response(pred.to("cpu").detach())

        return result_data

