import torch
from torch import nn

class ModelTrainingRepository:
    def get_model(self, model_name, partition, compressionProps = None, use_compression = False):
        if model_name == 'VGG':
            if partition == "server":
                from .vgg import VGG_server
                model = VGG_server(compressionProps, use_compression=use_compression)
            elif partition == "client":
                from .vgg import VGG_client
                model = VGG_client(compressionProps, use_compression=use_compression)
            else:
                from .vgg import VGG_unsplit
                model = VGG_unsplit()

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        elif model_name == 'Small':
            if partition == "server":
                from .small import Small_server
                model = Small_server()
            elif partition == "client":
                from .small import Small_client
                model = Small_client()
            else:
                from .small import Small_unsplit
                model = Small_unsplit()

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        else:
            raise f'No model with the name {model_name} was found in the repository'

        return model, loss_fn, optimizer

