import asyncio
import logging
from time import time

from sparse_framework import SparseNode

from .utils import count_model_parameters
from .model_meta_data import ModelMetaData
from .protocols import ParameterClientProtocol

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
        self.inputs = {}

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

    def buffer_input(self, model_meta_data, input_tensor):
        self.inputs[model_meta_data.model_id].append(self.transferToDevice(input_tensor))

    def pop_input(self, model_meta_data : ModelMetaData):
        return self.inputs[model_meta_data.model_id].pop(0)

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

    def result_received(self, result, callback):
        transferred_result = self.transferToHost(result["pred"])

        callback(transferred_result)

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
        self.inputs[model_meta_data.model_id] = []
        callback(load_task)

    async def save_model(self, model_meta_data : ModelMetaData):
        model = self.models[model_meta_data.model_id]['model']
#        await self.model_loader.save_model(model, model_meta_data)
