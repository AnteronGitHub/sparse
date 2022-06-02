from torch.autograd import Variable

from ..roles.task_executor import TaskExecutor

from .utils import get_device

from .serialization import decode_offload_request, encode_offload_response

class GradientCalculator(TaskExecutor):
    def __init__(self, model, loss_fn, optimizer):
        super().__init__()
        self.device = get_device()
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer

    def start(self):
        """Initialize executor by transferring the model to the processor memory."""
        super().start()
        self.logger.info(f"Task executor using {self.device} for processing")
        self.model.to(self.device)
        self.model.train()

    def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single gradient computation for the offloaded layers."""
        # Input de-serialization
        split_layer, labels = decode_offload_request(input_data)
        split_layer, labels = Variable(split_layer, requires_grad=True).to(
            self.device
        ), labels.to(self.device)
        split_layer.retain_grad()

        # Finish forward pass
        pred = self.model(split_layer)

        # Start back propagation
        loss = self.loss_fn(pred, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Result serialization
        return encode_offload_response(split_layer.grad.to("cpu").detach(), loss.item())

