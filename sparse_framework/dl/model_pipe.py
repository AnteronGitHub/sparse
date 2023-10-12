import asyncio
import logging
import pickle
from time import time

from sparse_framework import RXPipe

class ModelPipe(asyncio.Protocol):
    def __init__(self, task_executor, model_repository):
        self.logger = logging.getLogger("sparse")
        self.task_executor = task_executor
        self.model_repository = model_repository

    def model_loaded(self, load_task, split_layer, labels):
        model, loss_fn, optimizer = load_task.result()
        task_data = {
                'activation': split_layer,
                'labels': labels,
                'model': model,
                'loss_fn': loss_fn,
                'optimizer': optimizer
        }
        self.task_executor.submit_task(task_data, self.send_result)

    def send_result(self, result):
        self.transport.write(pickle.dumps(result))
        self.transport.close()

        latency = time() - self.received_at
        self.logger.info(f"Processed request in {latency} seconds.")

    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        self.transport = transport
        self.logger.debug(f"Received connection from {peername}.")

    def data_received(self, data):
        self.received_at = time()
        input_data = pickle.loads(data)
        split_layer, labels, model_meta_data = input_data['activation'], \
                                               input_data['labels'], \
                                               input_data['model_meta_data']

        load_task = self.model_repository.get_load_task(model_meta_data)
        if load_task is None:
            self.model_repository.load_model(model_meta_data,
                                             lambda task: self.model_loaded(task, split_layer, labels))
        elif not load_task.done():
            load_task.add_done_callback(lambda task: self.model_loaded(task, split_layer, labels))
        else:
            self.model_loaded(load_task, split_layer, labels)
