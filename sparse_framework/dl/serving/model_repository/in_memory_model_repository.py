import asyncio

from sparse_framework import Node

from ...models import ModuleQueue
from ...utils import count_model_parameters
from ..tcp_model_loader import TCPModelLoader
from ..model_meta_data import ModelMetaData
from .base_model_repository import BaseModelRepository

class InMemoryModelRepository(BaseModelRepository):
    def __init__(self, node : Node, device : str):
        self.node = node
        self.logger = self.node.logger
        self.model_loader = TCPModelLoader(self.node.config_manager.model_server_address,
                                           self.node.config_manager.model_server_port)

        self.device = device
        self.models = {}

    async def get_model(self, model_meta_data : ModelMetaData):
        if model_meta_data.model_id not in self.models.keys():
            self.logger.info(f"Downloading model '{model_meta_data.model_name}'")
            load_task = asyncio.create_task(self._load_model(model_meta_data))
            self.models[model_meta_data.model_id] = { 'model_name': model_meta_data.model_name, 'load_task': load_task }
            await load_task
        elif not self.models[model_meta_data.model_id]['load_task'].done():
            self.logger.info(f"Waiting for model '{model_meta_data.model_name}' to be downloaded")
            await asyncio.wait([self.models[model_meta_data.model_id]['load_task']])

        model_data = self.models[model_meta_data.model_id]
        return model_data['model'], model_data['loss_fn'], model_data['optimizer']

    async def _load_model(self, model_meta_data : ModelMetaData):
        """Loads a node over the network and into the executor device memory."""

        model, loss_fn, optimizer = await self.model_loader.load_model(model_meta_data)
        model = model.to(self.device)

        num_parameters = count_model_parameters(model)
        self.logger.info(f"Downloaded model '{model_meta_data.model_name}' with {num_parameters} parameters")

        self.models[model_meta_data.model_id]['model'] = model
        self.models[model_meta_data.model_id]['loss_fn'] = loss_fn
        self.models[model_meta_data.model_id]['optimizer'] = optimizer

    async def save_model(self, model : ModuleQueue, model_meta_data : ModelMetaData):
        await self.model_loader.save_model(model, model_meta_data)
