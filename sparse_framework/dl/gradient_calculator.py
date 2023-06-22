from torch.autograd import Variable
import torch

from .model_executor import ModelExecutor

from .serialization import decode_offload_request, encode_offload_request, decode_offload_response, encode_offload_response

class GradientCalculator(ModelExecutor):
    def start(self):
        super().start()
        self.model.train()

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single gradient computation for the offloaded layers."""
        split_layer, labels = decode_offload_request(input_data)
        split_layer = Variable(split_layer, requires_grad=True).to(self.device)
        split_layer.retain_grad()

        pred = self.model(split_layer)

        if self.task_deployer:
            input_data = encode_offload_request(pred.to("cpu").detach(), labels.to("cpu"))
            result_data = await self.task_deployer.deploy_task(input_data)
            split_grad, loss = decode_offload_response(result_data)
            split_grad = split_grad.to(self.device)

            self.optimizer.zero_grad()
            pred.backward(split_grad)
            self.optimizer.step()
        else:
            loss = self.loss_fn(pred, labels.to(self.device))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss = loss.item()

        result_data = encode_offload_response(split_layer.grad.to("cpu").detach(), loss)

        return result_data

