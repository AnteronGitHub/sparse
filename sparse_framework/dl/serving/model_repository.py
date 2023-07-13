class ModelRepository():
    def get_model(self, model_name : str, partition : str):
        pass

    def save_model(self, model_name : str, partition : str):
        pass

import os
import torch
from torch import nn

class ModelTrainingRepository(ModelRepository):
    def __init__(self, data_path : str = "/data/models"):
        self.data_path = data_path

    def get_model_state_filepath(self, model_name : str, partition : str):
        return os.path.join(self.data_path, f"{model_name}_{partition}.pt")

    def get_model(self, model_name, partition):
        state_path = self.get_model_state_filepath(model_name, partition)
        if not os.path.exists(state_path):
            state_path = None

        if model_name == 'VGG':
            if partition == "server":
                from sparse_framework.dl.models import VGG_server
                model = VGG_server(state_path = state_path)
            elif partition == "client":
                from sparse_framework.dl.models import VGG_client
                model = VGG_client(state_path = state_path)
            else:
                from sparse_framework.dl.models import VGG_unsplit
                model = VGG_unsplit()

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        elif model_name == 'Small':
            if partition == "server":
                from sparse_framework.dl.models import Small_server
                model = Small_server()
            elif partition == "client":
                from sparse_framework.dl.models import Small_client
                model = Small_client()
            else:
                from sparse_framework.dl.models import Small_unsplit
                model = Small_unsplit()

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        else:
            raise f"No model with the name {model_name} was found in the repository"

        return model, loss_fn, optimizer

    def save_model(self, model, model_name : str, partition : str):
        os.makedirs(self.data_path, exist_ok=True)
        torch.save(model.state_dict(), self.get_model_state_filepath(model_name, partition))
