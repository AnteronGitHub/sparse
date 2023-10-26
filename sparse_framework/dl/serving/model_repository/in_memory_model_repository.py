import asyncio
import logging

from torch.autograd import Variable

from sparse_framework import Node

from ...models import ModuleQueue
from ...utils import count_model_parameters
from ..tcp_model_loader import TCPModelLoader
from ..model_meta_data import ModelMetaData
from .base_model_repository import BaseModelRepository

class InMemoryModelRepository(BaseModelRepository):
    def __init__(self, node : Node, device : str):
        self.node = node
        self.logger = logging.getLogger("sparse")
        self.model_loader = TCPModelLoader(node.config_manager.model_server_address,
                                           node.config_manager.model_server_port)

        self.device = device
        self.models = {}

    def transferToDevice(self, tensor):
        return tensor.to(self.device)

    def transferToHost(self, tensor):
        return tensor.to("cpu")

    def get_load_task(self, model_meta_data : ModelMetaData):
        if model_meta_data.model_id not in self.models.keys():
            return None
        else:
            return self.models[model_meta_data.model_id]['load_task']

    def load_model(self, model_meta_data : ModelMetaData, callback):
        load_task = asyncio.create_task(self.model_loader.load_model(model_meta_data))
        load_task.add_done_callback(lambda task: self.model_loaded(model_meta_data, task, callback))
        self.models[model_meta_data.model_id] = { "model_meta_data": model_meta_data,
                                                  "load_task": load_task }

        self.logger.info(f"Loading model '{model_meta_data.model_name}'.")
        return load_task

    def model_loaded(self, model_meta_data : ModelMetaData, load_task, callback):
        model, loss_fn, optimizer = load_task.result()
        self.models[model_meta_data.model_id]["model"] = model.to(self.device)
        self.models[model_meta_data.model_id]["loss_fn"] = loss_fn
        self.models[model_meta_data.model_id]["optimizer"] = optimizer
        callback(load_task)

    async def save_model(self, model_meta_data : ModelMetaData):
        model = self.models[model_meta_data.model_id]['model']
        await self.model_loader.save_model(model, model_meta_data)
