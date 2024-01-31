import asyncio
from time import time

from sparse_framework import TaskExecutor
from . import count_model_parameters, get_device

__all__ = ["TensorExecutor"]

class TensorExecutor(TaskExecutor):
    def __init__(self, use_batching : bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = get_device()
        self.use_batching = use_batching
        self.batch_no = 0

    async def start(self):
        self.logger.info(f"Task executor using {self.device} for tensor processing (Batching: {self.use_batching}).")
        await super().start()

    def execute_task(self, fn_name, input_data, callback, lock):
        if fn_name == "forward_propagate":
            self.forward_propagate(input_data, callback, lock)
        elif fn_name == "backward_propagate":
            self.backward_propagate(input_data, callback, lock)
        else:
            self.logger.debug(f"Received unknown function '{fn_name}' call.")

    def forward_propagate(self, model_meta_data, callback, lock):
        """Run forward pass for specified model with specified input tensor."""
        model = self.memory_buffer.get_model(model_meta_data)

        if self.use_batching:
            features, callbacks, statistics_records = self.memory_buffer.dispatch_batch(model_meta_data, lock) 
        else:
            features, callbacks, statistics_records = self.memory_buffer.pop_input(model_meta_data, lock) 

        task_started_at = time()
        pred = model(features)
        task_completed_at = time()

        for record in statistics_records:
            record.task_started(task_started_at, self.batch_no)
            record.task_completed(task_completed_at)

        self.batch_no += 1
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
