import asyncio
import torch

from sparse_framework import RXPipe

class BatchedRXPipe(RXPipe):
    def __init__(self, **args):
        super().__init__(**args)
        self.active_tasks = set()
        self.pending_tasks = {}
        self.batches = {}

    async def dispatch_batch(self, model_meta_data):
        input_batch = self.batches[model_meta_data.model_id]
        no_dispatched_tasks = len(input_batch["layers"])
        self.logger.info(f"Dispatched a batch of {no_dispatched_tasks} tasks.")
        self.batches.__delitem__(model_meta_data.model_id)
        input_data = {
                "activation": torch.cat(input_batch["layers"]),
                "labels": torch.cat(input_batch["labels"]),
                "model_meta_data": input_batch["model_meta_data"],
                "capacity": input_batch["capacity"]
                }
        return await self.task_executor.execute_task(input_data)

    async def handle_request(self, input_data : dict, context : dict) -> dict:
        if len(self.active_tasks) == 0:
            single_job = asyncio.create_task(self.task_executor.execute_task(input_data))
            self.active_tasks.add(single_job)
            single_job.add_done_callback(self.active_tasks.discard)
            self.logger.debug("Created a single job.")

            output_data = await single_job
        else:
            split_layer, labels, model_meta_data, client_capacity = input_data['activation'], \
                                                                    input_data['labels'], \
                                                                    input_data['model_meta_data'], \
                                                                    input_data['capacity']
            if model_meta_data.model_id not in self.batches.keys():
                batch_job = asyncio.create_task(self.dispatch_batch(model_meta_data))
                self.active_tasks.add(batch_job)
                batch_job.add_done_callback(self.active_tasks.discard)
                self.batches[model_meta_data.model_id] = { "layers": [split_layer],
                                                           "labels": [labels],
                                                           "model_meta_data": model_meta_data,
                                                           "job": batch_job,
                                                           "capacity": client_capacity }
                batch_index = 0
                self.logger.debug(f"Created a batch job.")

                output_data = await batch_job
            else:
                batch = self.batches[model_meta_data.model_id]
                batch_index = len(batch["layers"])
                batch["layers"].append(split_layer)
                batch["labels"].append(labels)
                self.logger.debug("Added request to an existing batch job.")

                output_data = await asyncio.wait_for(batch["job"], timeout=120)

        return output_data, context

