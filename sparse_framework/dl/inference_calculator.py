
from ..task_executor import TaskExecutor

from .serialization import decode_offload_inference_request, \
                           encode_offload_inference_request, \
                           decode_offload_inference_response, \
                           encode_offload_inference_response
from .utils import get_device

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
        split_layer = decode_offload_inference_request(input_data).to(self.device)
        pred = self.model(split_layer)

        if self.task_deployer:
            self.logger.debug("Deploying to the next worker further")

            offload_input_data = encode_offload_inference_request(pred.to("cpu").detach())
            result_data = await self.task_deployer.deploy_task(offload_input_data)
        else:
            self.logger.debug("Not deploying task any further")
            result_data = encode_offload_inference_response(pred.to("cpu").detach())

        return result_data

