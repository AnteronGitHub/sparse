import asyncio
from time import time

from sparse_framework import TaskExecutor
from . import count_model_parameters, get_device

__all__ = ["TensorExecutor"]

class TensorExecutor(TaskExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = get_device()

    async def start(self):
        self.logger.info(f"Task executor using {self.device} for tensor processing.")
        await super().start()

    def execute_task(self, fn_name, input_data, callback):
        if fn_name == "forward_propagate":
            self.forward_propagate(input_data, callback)
        elif fn_name == "backward_propagate":
            self.backward_propagate(input_data, callback)
        else:
            self.logger.debug(f"Received unknown function '{fn_name}' call.")

    def forward_propagate(self, model_meta_data, callback):
        """Run forward pass for specified model with specified input tensor."""
        model = self.memory_buffer.get_model(model_meta_data)
        features, callbacks, statistics_records = self.memory_buffer.dispatch_batch(model_meta_data)

        task_started_at = time()
        pred = model(features)
        task_completed_at = time()

        for record in statistics_records:
            record.task_started(task_started_at)
            record.task_completed(task_completed_at)

        callback(pred, callbacks)

    def backward_propagate(self, input_data: dict, callback) -> dict:
        split_layer, labels, model, loss_fn, optimizer, pred = input_data['activation'], \
                                                               input_data['labels'], \
                                                               input_data['model'], \
                                                               input_data['loss_fn'], \
                                                               input_data['optimizer'], \
                                                               input_data['pred']

        loss = loss_fn(pred, labels.to(self.device))

#        optimizer.zero_grad()
        loss.backward()
#        optimizer.step()

        callback({ "gradient": split_layer.grad, "loss": loss.item() })

    def backpropagate_split(self, response_data, input_data: dict, callback) -> dict:
        split_layer, labels, model, optimizer = input_data['activation'], \
                                                input_data['labels'], \
                                                input_data['model'], \
                                                input_data['optimizer']
        split_grad, loss = response_data['gradient'], response_data['loss']

        split_grad = split_grad.to(self.device)
        optimizer.zero_grad()
        pred.backward(split_grad)
        optimizer.step()

        callback({ "gradient": split_layer.grad.to("cpu").detach(), "loss": loss })
