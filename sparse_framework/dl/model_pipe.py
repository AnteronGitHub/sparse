import asyncio

from sparse_framework import RXPipe
from sparse_framework.dl import InMemoryModelRepository, get_device

class ModelPipe(RXPipe):
    def __init__(self, **args):
        super().__init__(**args)

        self.models = {}
        self.model_repository = None

    def set_node(self, node):
        super().set_node(node)
        self.model_repository = InMemoryModelRepository(self.node, get_device())

    def create_load_task(self, model_meta_data):
        load_task = asyncio.create_task(self.model_repository.get_model(model_meta_data))
        self.models[model_meta_data.model_id] = { "model_meta_data": model_meta_data,
                                                  "load_task": load_task }
        self.logger.debug("Created a model load task for a request.")
        return load_task

    async def handle_request(self, input_data : dict, context : dict) -> dict:
        split_layer, labels, model_meta_data = input_data['activation'], \
                                               input_data['labels'], \
                                               input_data['model_meta_data']

        if model_meta_data.model_id not in self.models.keys():
            load_task = self.create_load_task(model_meta_data)
            await load_task
        else:
            load_task = self.models[model_meta_data.model_id]['load_task']

        if not load_task.done():
            self.logger.debug("Waiting for a model load to complete.")
            await asyncio.wait_for(load_task, timeout=120)

        model, loss_fn, optimizer = load_task.result()

        task_data = {
                'activation': split_layer,
                'labels': labels,
                'model': model,
                'loss_fn': loss_fn,
                'optimizer': optimizer
        }
        output_data = await self.task_executor.execute_task(task_data)

        return output_data, context

