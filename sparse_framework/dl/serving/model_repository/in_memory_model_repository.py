from sparse_framework import Node

from ..tcp_model_loader import TCPModelLoader
from ..model_meta_data import ModelMetaData
from .base_model_repository import BaseModelRepository

from ...utils import count_model_parameters

class InMemoryModelRepository(BaseModelRepository):
    def __init__(self, node : Node, device : str):
        self.node = node
        self.logger = self.node.logger
        self.model_loader = TCPModelLoader(self.node.config_manager.model_server_address,
                                           self.node.config_manager.model_server_port)

        self.device = device
        self.models = {}

    async def get_model(self, model_meta_data : ModelMetaData):
        if model_meta_data.model_name not in self.models.keys():
            await self._load_model(model_meta_data)

        [model, loss_fn, optimizer] = self.models[model_meta_data.model_name]
        return model, loss_fn, optimizer

    async def _load_model(self, model_meta_data : ModelMetaData):
        """Loads a node over the network and into the executor device memory."""

        model, loss_fn, optimizer = await self.model_loader.load_model(model_meta_data)
        model = model.to(self.device)

        num_parameters = count_model_parameters(model)
        self.logger.info(f"Downloaded model '{model_meta_data.model_name}' with {num_parameters}'")

        self.models[model_meta_data.model_name] = [model, loss_fn, optimizer]
