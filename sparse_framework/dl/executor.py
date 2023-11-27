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

    def execute_task(self, fn_name, input_data: dict, callback) -> dict:
        if fn_name == "forward_propagate":
            self.forward_propagate(input_data, callback)
        elif fn_name == "backward_propagate":
            self.backward_propagate(input_data, callback)
        else:
            self.logger.debug(f"Received unknown function '{fn_name}' call.")

    def forward_propagate(self, input_data: dict, callback) -> dict:
        """Run forward pass for specified model with specified input tensor."""
        model_meta_data, statistics_record = input_data['model_meta_data'], input_data['statistics_record']
        statistics_record.task_started()

        model = self.memory_buffer.get_model(model_meta_data)
        features = self.memory_buffer.pop_input(model_meta_data)
        pred = model(features)

        statistics_record.task_completed()

        callback({ "pred": pred })

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
