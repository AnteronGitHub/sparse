import asyncio
import logging
import pickle
from time import time

from sparse_framework.stats import ServerRequestStatistics

class ModelServeServerProtocol(asyncio.Protocol):
    def __init__(self, node, queue, stats_queue):
        self.logger = logging.getLogger("sparse")
        self.queue = queue
        self.stats_queue = stats_queue
        self.model_repository = node.get_model_repository()

        self.model_meta_data = None
        self.statistics = ServerRequestStatistics(node.node_id, stats_queue)

    def initialize_stream(self, input_data):
        self.statistics.task_started("initialize_stream")
        self.model_meta_data = input_data['model_meta_data']
        load_task = self.model_repository.get_load_task(self.model_meta_data)
        if load_task is None:
            self.model_repository.load_model(self.model_meta_data, self.model_loaded)
        elif not load_task.done():
            load_task.add_done_callback(self.model_loaded)
        else:
            self.model_loaded(load_task)

        self.statistics.current_record.queued()

    def model_loaded(self, load_task):
        self.send_result({ "statusCode": 200 })

    def offload_task(self, input_data):
        self.statistics.task_started("offload_task")

        load_task = self.model_repository.get_load_task(self.model_meta_data)
        model, loss_fn, optimizer = load_task.result()
        task_data = {
                'activation': self.model_repository.transferToDevice(input_data['activation']),
                'model': model
        }

        self.statistics.current_record.queued()
        self.queue.put_nowait(("forward_propagate", task_data, self.forward_propagated))

    def forward_propagated(self, result):
        task_latency = result["latency"]
        self.statistics.current_record.set_task_latency(task_latency)

        self.send_result({ "pred": self.model_repository.transferToHost(result["pred"]) })

    def send_result(self, result):
        self.transport.write(pickle.dumps(result))

        self.statistics.task_completed()

    def connection_made(self, transport):
        self.statistics.connected()
        peername = transport.get_extra_info('peername')
        self.transport = transport
        self.logger.info(f"Received connection from {peername}.")

    def data_received(self, data):
        try:
            input_data = pickle.loads(data)
        except pickle.UnpicklingError:
            self.logger.error("Unpickling error occurred")
            # TODO: implement better handling
            self.send_result({ "pred": None })
            return

        if input_data["op"] == "initialize_stream":
            self.initialize_stream(input_data)
        else:
            self.offload_task(input_data)

    def connection_lost(self, exc):
        self.logger.info(self.statistics)
