from ...models import ModuleQueue
from ..model_meta_data import ModelMetaData

class BaseModelRepository():
    async def get_model(self, model_meta_data : ModelMetaData):
        pass

    def save_model(self, model : ModuleQueue, model_meta_data : ModelMetaData):
        pass
