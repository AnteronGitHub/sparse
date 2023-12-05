import os
import torch
from torch import nn

from .models import ModuleQueue
from .model_meta_data import ModelMetaData

__all__ = [ "BaseModelRepository",
            "DiskModelRepository" ]

class BaseModelRepository():
    async def get_model(self, model_meta_data : ModelMetaData):
        pass

    async def save_model(self, model : ModuleQueue, model_meta_data : ModelMetaData):
        pass

class DiskModelRepository(BaseModelRepository):
    def __init__(self, data_path : str = "/data/models"):
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

    def get_model(self, model_meta_data : ModelMetaData):
        if model_meta_data.model_name == 'VGG':
            from sparse_framework.dl.models import VGG_unsplit
            model = VGG_unsplit(state_path = self.data_path)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        elif model_meta_data.model_name == 'Small':
            from sparse_framework.dl.models import Small_unsplit
            model = Small_unsplit()

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        else:
            raise f"No model with the name {model_name} was found in the repository"

        return model, loss_fn, optimizer

    def save_model(self, model : ModuleQueue, model_meta_data : ModelMetaData):
        model.save_parameters(self.data_path, model_meta_data.model_name)

