import asyncio
import logging
from time import time

import torch

from sparse_framework import SparseNode

from .utils import count_model_parameters
from .model_meta_data import ModelMetaData
from .protocols import ParameterClientProtocol

class TaskData:
    def __init__(self, input_data, done_callback, statistics_record):
        self.input_data = input_data
        self.done_callback = done_callback
        self.statistics_record = statistics_record

class MemoryBuffer:
    """Memory buffer for Node's task executor.

    Buffer ensures that when a task is created for the executor, all of the needed data are available in the executor
    device memory. After the executor task has been processed, memory buffer transfers the data to the network device
    for result transmission.
    """
    def __init__(self, node : SparseNode, device : str):
        self.logger = logging.getLogger("sparse")
        self.node = node
        self.device = device

        self.models = {}
        self.task_data_buffer = {}

    def transferToDevice(self, tensor):
        return tensor.to(self.device)

    def transferToHost(self, tensor):
        return tensor.to("cpu")

    def get_load_task(self, model_meta_data : ModelMetaData):
        if model_meta_data.model_id not in self.models.keys():
            return None
        else:
            return self.models[model_meta_data.model_id]['load_task']

    def get_model(self, model_meta_data : ModelMetaData):
        load_task = self.get_load_task(model_meta_data)
        return load_task.result()

    def buffer_input(self, model_meta_data, input_tensor, rx_callback, statistics_record) -> int:
        """Appends an input tensor to the specified model's input buffer and returns its index.
        """
        index = len(self.task_data_buffer[model_meta_data.model_id])
        callback = lambda result: self.result_received(result, rx_callback, index)
        task_data = TaskData(self.transferToDevice(input_tensor), callback, statistics_record)
        self.task_data_buffer[model_meta_data.model_id].append(task_data)
        return index

    def pop_input(self, model_meta_data : ModelMetaData):
        return self.task_data_buffer[model_meta_data.model_id].pop(0)

    def dispatch_batch(self, model_meta_data : ModelMetaData):
        input_data = []
        callbacks = []
        statistics_records = []
        for task_data in self.task_data_buffer[model_meta_data.model_id]:
            input_data.append(task_data.input_data)
            callbacks.append(task_data.done_callback)
            statistics_records.append(task_data.statistics_record)

        self.task_data_buffer[model_meta_data.model_id] = []

        return torch.cat(input_data), callbacks, statistics_records

    def start_stream(self, model_meta_data, callback):
        """Initializes a new offloading stream by ensuring that the model parameters are available locally.
        """
        load_task = self.get_load_task(model_meta_data)
        if load_task is None:
            self.load_model(model_meta_data, callback)
        elif not load_task.done():
            load_task.add_done_callback(callback)
        else:
            callback(load_task)

    def result_received(self, result, callback, index):
        transferred_result = self.transferToHost(result)

        result = transferred_result[index]

        callback(result)

    def load_model(self, model_meta_data : ModelMetaData, callback):
        model_loader_protocol_factory = lambda on_con_lost, stats_queue: \
                                            lambda: ParameterClientProtocol(model_meta_data, on_con_lost)
        load_task = asyncio.create_task(self.node.connect_to_server(model_loader_protocol_factory, \
                                                                    self.node.config.model_server_address, \
                                                                    self.node.config.model_server_port))
        load_task.add_done_callback(lambda task: self.model_loaded(model_meta_data, task, callback))
        self.models[model_meta_data.model_id] = { "model_meta_data": model_meta_data,
                                                  "load_task": load_task }

        self.logger.info(f"Loading model '{model_meta_data.model_name}'.")
        return load_task

    def model_loaded(self, model_meta_data : ModelMetaData, load_task, callback):
        model = load_task.result()
        self.models[model_meta_data.model_id]["model"] = model.to(self.device)
        self.task_data_buffer[model_meta_data.model_id] = []
        callback(load_task)

    async def save_model(self, model_meta_data : ModelMetaData):
        model = self.models[model_meta_data.model_id]['model']
#        await self.model_loader.save_model(model, model_meta_data)
