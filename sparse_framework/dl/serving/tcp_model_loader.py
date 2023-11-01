from torch import nn

from sparse_framework.networking import TCPClient

from .model_meta_data import ModelMetaData

class TCPModelLoader(TCPClient):
    async def load_model(self, model_meta_data : ModelMetaData):
        response_data = await self._create_request({ 'method' : 'get_model',
                                                     'model_meta_data': model_meta_data })
        return response_data['model'], response_data['loss_fn'], response_data['optimizer']

    async def save_model(self, model : nn.Module, model_meta_data : ModelMetaData):
        response_data = await self._create_request({ 'method' : 'save_model',
                                                     'model' : model,
                                                     'model_meta_data': model_meta_data })
        return response_data['status']
