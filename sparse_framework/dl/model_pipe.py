import asyncio
import logging
import pickle
from time import time

from sparse_framework import RXPipe

class ModelPipe(asyncio.Protocol):
    def __init__(self, queue, task_executor, model_repository):
        self.logger = logging.getLogger("sparse")
        self.queue = queue
        self.task_executor = task_executor
        self.model_repository = model_repository

        self.model_meta_data = None

    def initialize_stream(self, input_data):
        self.model_meta_data = input_data['model_meta_data']
        load_task = self.model_repository.get_load_task(self.model_meta_data)
        if load_task is None:
            self.model_repository.load_model(self.model_meta_data, self.model_loaded)
        elif not load_task.done():
            load_task.add_done_callback(self.model_loaded)
        else:
            self.model_loaded(load_task)

    def model_loaded(self, load_task):
        self.send_result({ "statusCode": 200 })

    def offload_task(self, input_data):
        split_layer = input_data['activation']
        load_task = self.model_repository.get_load_task(self.model_meta_data)
        model, loss_fn, optimizer = load_task.result()
        task_data = {
                'activation': split_layer,
                'model': model
        }
        self.queue.put_nowait(("forward_propagate", task_data, self.forward_propagated))

    def forward_propagated(self, result):
        self.send_result({ "pred": result["pred"] }, task_latency=result["latency"])

    def send_result(self, result, task_latency=0):
        self.transport.write(pickle.dumps(result))

        latency = time() - self.received_at
        self.logger.info(f"E2E lat./Task lat./Ratio: {1000.0*latency:.2f} ms / {1000.0*task_latency:.2f} ms / {100.0*task_latency/latency:.2f} %.")

    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        self.transport = transport
        self.logger.info(f"Received connection from {peername}.")

    def data_received(self, data):
        self.logger.debug(f"Received request.")
        self.received_at = time()

        try:
            input_data = pickle.loads(data)
        except pickle.UnpicklingError:
            self.logger.error("Unpickling error occurred")
            self.send_result({ "pred": None })
            return

        if input_data["op"] == "initialize_stream":
            self.initialize_stream(input_data)
        else:
            self.offload_task(input_data)
