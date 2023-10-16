import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

from torch.autograd import Variable

from sparse_framework import TaskExecutor
from sparse_framework.dl import count_model_parameters, get_device, ModelExecutor

class GradientCalculator(TaskExecutor):
    def __init__(self, **args):
        super().__init__(**args)
        self.device = get_device()
        self.executor = ThreadPoolExecutor()

    async def start(self, queue):
        self.logger.info(f"Task executor using {self.device} for tensor processing.")

        loop = asyncio.get_running_loop()
        while True:
            input_data, callback = await queue.get()
            await loop.run_in_executor(self.executor, functools.partial(self.execute_task, input_data, callback))
            queue.task_done()

    def execute_task(self, input_data: dict, callback) -> dict:
        """Execute a single gradient computation for the offloaded layers."""
        split_layer, labels, model, loss_fn, optimizer = input_data['activation'], \
                                                         input_data['labels'], \
                                                         input_data['model'], \
                                                         input_data['loss_fn'], \
                                                         input_data['optimizer']

        split_layer = Variable(split_layer, requires_grad=True).to(self.device)
        split_layer.retain_grad()

        pred = model(split_layer)

        self.backpropagate(pred, input_data, callback)

    def backpropagate(self, pred, input_data: dict, callback) -> dict:
        split_layer, labels, model, loss_fn, optimizer = input_data['activation'], \
                                                         input_data['labels'], \
                                                         input_data['model'], \
                                                         input_data['loss_fn'], \
                                                         input_data['optimizer']

        loss = loss_fn(pred, labels.to(self.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()

        gradient = None if split_layer.grad is None else split_layer.grad.to("cpu").detach()

        callback({ "gradient": gradient, "loss": loss })

    def backpropagate_split(self, response_data, input_data: dict, callback) -> dict:
        split_layer, labels, model, loss_fn, optimizer = input_data['activation'], \
                                                         input_data['labels'], \
                                                         input_data['model'], \
                                                         input_data['loss_fn'], \
                                                         input_data['optimizer']
        split_grad, loss = response_data['gradient'], response_data['loss']

        split_grad = split_grad.to(self.device)
        optimizer.zero_grad()
        pred.backward(split_grad)
        optimizer.step()

        callback({ "gradient": gradient, "loss": loss })
