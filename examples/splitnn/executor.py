import torch
from time import time

from sparse_framework import TaskExecutor
from utils import count_model_parameters, get_device
from vgg import VGG_unsplit

__all__ = ["TensorExecutor"]

class TensorExecutor(TaskExecutor):
    def __init__(self, use_batching : bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = get_device()
        self.use_batching = use_batching
        self.batch_no = 0

        self.model = None

    async def start(self):
        self.model = VGG_unsplit()
        self.logger.info(f"Serving inference for VGG using {self.device} (Batching: {self.use_batching}).")
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
        if self.use_batching:
            features, callbacks, statistics_records = self.memory_buffer.dispatch_batch(lock)
            features = torch.cat(features)
        else:
            features, callbacks, statistics_records = self.memory_buffer.pop_input(lock)

        task_started_at = time()
        pred = self.model(features)
        task_completed_at = time()

        for record in statistics_records:
            record.task_started(task_started_at, self.batch_no)
            record.task_completed(task_completed_at)

        self.batch_no += 1
        callback(pred, callbacks)

