import asyncio
from torch.autograd import Variable
from torch.nn import Module

from ..task_executor import TaskExecutor

from .serialization import decode_offload_request, encode_offload_request, decode_offload_response, encode_offload_response
from .utils import get_device

class GradientCalculator(TaskExecutor):
    def __init__(self, model : Module, loss_fn, optimizer):
        super().__init__()
        self.device = get_device()
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer

    def start(self):
        """Initialize executor by transferring the model to the processor memory."""
        super().start()
        num_parameters = 0
        for param in self.model.parameters():
            num_parameters += param.nelement()
        self.logger.info(f"Training {type(self.model).__name__} model with {num_parameters} parameters using {self.device} for processing")
        self.model.to(self.device)
        self.model.train()

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single gradient computation for the offloaded layers."""
        # Input de-serialization
        split_layer, labels = decode_offload_request(input_data)
        split_layer, labels = Variable(split_layer, requires_grad=True).to(
            self.device
        ), labels.to(self.device)
        split_layer.retain_grad()

        # Local forward pass
        pred = self.model(split_layer)

        if self.task_deployer:
            self.logger.debug("Deploying to the next worker further")

            # Offloaded layers
            input_data = encode_offload_request(pred.to("cpu").detach(), labels.to("cpu"))
            result_data = await self.task_deployer.deploy_task(input_data)

            # Local back propagation
            split_grad, loss = decode_offload_response(result_data)
            split_grad = split_grad.to(self.device)
            self.optimizer.zero_grad()
            pred.backward(split_grad)
            self.optimizer.step()
        else:
            self.logger.debug("Not deploying task any further")
            # Start back propagation
            loss = self.loss_fn(pred, labels)
            self.logger.debug("Computed loss")
            self.optimizer.zero_grad()
            loss.backward()
            self.logger.debug("Updated parameters")
            self.optimizer.step()
            self.logger.debug("Updated optimizer")
            loss = loss.item()

        # Result serialization
        result_data = encode_offload_response(split_layer.grad.to("cpu").detach(), loss)

        self.logger.debug("Executed task")
        return result_data

