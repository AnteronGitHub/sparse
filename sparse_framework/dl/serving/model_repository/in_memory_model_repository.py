import asyncio
import logging

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
        self.model_loader = TCPModelLoader(self.node.config_manager.model_server_address,
                                           self.node.config_manager.model_server_port)

        self.device = device
        self.models = {}

    async def get_model(self, model_meta_data : ModelMetaData):
        if model_meta_data.model_id not in self.models.keys():
            if hasattr(self.node, 'task_deployer'):
                load_task = asyncio.create_task(self._alloc_model(model_meta_data))
            else:
                load_task = asyncio.create_task(self._load_model(model_meta_data))

            self.models[model_meta_data.model_id] = { 'model_name': model_meta_data.model_name,
                                                      'load_task': load_task }
            await load_task
        elif not self.models[model_meta_data.model_id]['load_task'].done():
            self.logger.info(f"Waiting for model '{model_meta_data.model_name}' to be downloaded")
            await asyncio.wait([self.models[model_meta_data.model_id]['load_task']])

        model_data = self.models[model_meta_data.model_id]
        return model_data['model'], model_data['loss_fn'], model_data['optimizer']

    async def _alloc_model(self, model_meta_data : ModelMetaData):
        """Allocates an empty module queue instance into the executor device memory."""

        self.logger.debug(f"Allocating '{model_meta_data.model_name}' model instance")
        model = ModuleQueue()
        loss_fn = None
        optimizer = None
        model = model.to(self.device)

        self.models[model_meta_data.model_id]['model'] = model
        self.models[model_meta_data.model_id]['loss_fn'] = loss_fn
        self.models[model_meta_data.model_id]['optimizer'] = optimizer

        self.logger.info(f"Allocated an empty '{model_meta_data.model_name}' model instance")

    async def _load_model(self, model_meta_data : ModelMetaData):
        """Loads a model over the network and into the executor device memory."""

        self.logger.debug(f"Downloading model '{model_meta_data.model_name}'")
        model, loss_fn, optimizer = await self.model_loader.load_model(model_meta_data)
        model = model.to(self.device)

        self.models[model_meta_data.model_id]['model'] = model
        self.models[model_meta_data.model_id]['loss_fn'] = loss_fn
        self.models[model_meta_data.model_id]['optimizer'] = optimizer

        num_parameters = count_model_parameters(model)
        self.logger.info(f"Downloaded '{model_meta_data.model_name}' model instance with {num_parameters} parameters")

    async def save_model(self, model_meta_data : ModelMetaData):
        model = self.models[model_meta_data.model_id]['model']
        await self.model_loader.save_model(model, model_meta_data)
