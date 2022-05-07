import torch
from torch import nn
from torch.autograd import Variable

from serialization import decode_offload_request, encode_offload_response
from models.neural_network import NeuralNetwork_server
from roles.worker import Worker, TaskExecutor

class GradientCalculator(TaskExecutor):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = NeuralNetwork_server()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    def start(self):
        """Initialize executor by transferring the model to the processor memory.
        """
        self.model.to(self.device)
        self.model.train()

    def execute_task(self, input_data : bytes) -> bytes:
        """Execute a single gradient computation for the offloaded layers.
        """
        # Input de-serialization
        split_layer, labels = decode_offload_request(input_data)
        split_layer, labels = Variable(split_layer, requires_grad=True).to(self.device), labels.to(self.device)
        split_layer.retain_grad()

        # Finish forward pass
        pred = self.model(split_layer)

        # Start back propagation
        loss = self.loss_fn(pred, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Result serialization
        return encode_offload_response(split_layer.grad.to('cpu').detach(), loss.item())

class SplitTrainingServer(Worker):
    def __init__(self):
        super().__init__(task_executor = GradientCalculator())

    def start(self):
        self.task_executor.start()
        super().start()

if __name__ == "__main__":
    SplitTrainingServer().start()
