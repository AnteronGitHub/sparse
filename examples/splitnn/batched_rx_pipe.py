import asyncio
import torch

from sparse_framework.dl import ModelPipe

class BatchedRXPipe(ModelPipe):
    def __init__(self, **args):
        super().__init__(**args)

        self.batches = {}
        self.executor_tasks = {}

    def create_batch(self, model_meta_data, split_layer, labels, executor_task):
        self.batches[model_meta_data.model_id] = {
            "layers": [split_layer],
            "labels": [labels],
            "executor_task": executor_task
        }

        self.logger.debug(f"Created a batch.")

    def append_batch(self, model_meta_data, split_layer, labels):
        self.batches[model_meta_data.model_id]["layers"].append(split_layer)
        self.batches[model_meta_data.model_id]["labels"].append(labels)

        self.logger.debug("Added request to batch.")

    def dispatch_batch(self, model_meta_data):
        layers = self.batches[model_meta_data.model_id]["layers"]
        labels = self.batches[model_meta_data.model_id]["labels"]
        self.batches.pop(model_meta_data.model_id)

        no_dispatched_tasks = len(layers)
        self.logger.debug(f"Dispatched a batch .")

        return layers, labels

    async def create_executor_task(self, model_meta_data):
        if model_meta_data.model_id not in self.models.keys():
            load_task = self.create_load_task(model_meta_data)
            await load_task
        else:
            load_task = self.models[model_meta_data.model_id]['load_task']

        if not load_task.done():
            await asyncio.wait_for(load_task, timeout=120)

        model, loss_fn, optimizer = load_task.result()
        layers, labels = self.dispatch_batch(model_meta_data)

        task_data = {
                "activation": torch.cat(layers),
                "labels": torch.cat(labels),
                'model': model,
                'loss_fn': loss_fn,
                'optimizer': optimizer
        }

        executor_task = asyncio.create_task(self.task_executor.execute_task(task_data))
        return await executor_task

    async def handle_request(self, input_data : dict, context : dict) -> dict:
        split_layer, labels, model_meta_data, client_capacity = input_data['activation'], \
                                                                input_data['labels'], \
                                                                input_data['model_meta_data'], \
                                                                input_data['capacity']

        if model_meta_data.model_id not in self.batches.keys():
            executor_task = asyncio.create_task(self.create_executor_task(model_meta_data))
            self.create_batch(model_meta_data, split_layer, labels, executor_task)

            await executor_task
        else:
            self.append_batch(model_meta_data, split_layer, labels)
            executor_task = self.batches[model_meta_data.model_id]['executor_task']

        if not executor_task.done():
            await asyncio.wait_for(executor_task, timeout=120)

        return executor_task.result(), context

