import torch
from torch import nn

class ModelTrainingRepository:
    def get_model(self, model_name):
        if model_name == 'VGG_unsplit':
            from .vgg import VGG_unsplit
            model = VGG_unsplit()
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        elif model_name == 'VGG_client':
            from .vgg import VGG_client
            model = VGG_client()
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        elif model_name == 'VGG_server':
            from .vgg import VGG_server
            model = VGG_server()
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        elif model_name == 'Small_unsplit':
            from .small import Small_unsplit
            model = Small_unsplit()
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        elif model_name == 'Small_client':
            from .small import Small_client
            model = Small_client()
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        elif model_name == 'Small_server':
            from .small import Small_server
            model = Small_server()
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        else:
            raise f'No model with the name {model_name} was found in the repository'

        return model, loss_fn, optimizer

