import asyncio
from time import time

from torch.utils.data import DataLoader

from sparse_framework.networking.protocols import SparseProtocol
from sparse_framework.stats import ServerRequestStatistics, ClientRequestStatistics

__all__ = ["ModelServeClientProtocol", "ModelServeServerProtocol"]

TARGET_FPS = 5.0

class ModelServeClientProtocol(SparseProtocol):
    def __init__(self, data_source_id, dataset, model_meta_data, on_con_lost, no_samples, target_rate = 1/TARGET_FPS, stats_queue = None):
        super().__init__(on_con_lost)
        self.dataloader = DataLoader(dataset, 1)
        self.model_meta_data = model_meta_data
        self.no_samples = no_samples
        self.target_rate = target_rate
        self.statistics = ClientRequestStatistics(data_source_id, stats_queue)

    def start_stream(self):
        self.statistics.create_record("initialize_stream")
        self.send_payload({ 'op': "initialize_stream", 'model_meta_data': self.model_meta_data })
        self.statistics.request_sent()

    def stream_started(self, result_data):
        latency = self.statistics.task_completed()
        self.logger.info(f"Initialized stream in {latency:.2f} seconds with {1.0/self.target_rate:.2f} FPS target rate.")
        self.offload_task()

    def offload_task(self):
        self.statistics.create_record("offload_task")
        self.no_samples -= 1
        features, labels = next(iter(self.dataloader))
        self.send_payload({ 'op': 'offload_task',
                            'activation': features,
                            'labels': labels,
                            'model_meta_data': self.model_meta_data })

    def offload_task_completed(self, result_data):
        latency = self.statistics.task_completed()

        if (self.no_samples > 0):
            loop = asyncio.get_running_loop()
            loop.call_later(self.target_rate-latency if self.target_rate > latency else 0, self.offload_task)
        else:
            self.transport.close()

    def connection_made(self, transport):
        self.statistics.connected()

        super().connection_made(transport)

        self.start_stream()

    def payload_received(self, payload):
        if "statusCode" in payload.keys():
            self.stream_started(payload)
        else:
            self.offload_task_completed(payload)

    def send_payload(self, payload):
        super().send_payload(payload)

        self.statistics.request_sent()

    def connection_lost(self, exc):
        super().connection_lost(exc)

        self.logger.info(self.statistics)

class ModelServeServerProtocol(SparseProtocol):
    def __init__(self, node, queue, stats_queue):
        super().__init__()

        self.queue = queue
        self.stats_queue = stats_queue
        self.model_repository = node.get_model_repository()
        self.statistics = ServerRequestStatistics(node.node_id, stats_queue)

        self.model_meta_data = None

    def initialize_stream(self, input_data):
        self.statistics.create_record("initialize_stream")
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
        self.send_payload({ "statusCode": 200 })

    def offload_task(self, input_data):
        deserialization_started = time()

        load_task = self.model_repository.get_load_task(self.model_meta_data)
        model, loss_fn, optimizer = load_task.result()
        task_data = {
                'activation': self.model_repository.transferToDevice(input_data['activation']),
                'model': model
        }

        deserialization_latency = time() - deserialization_started

        self.queue.put_nowait(("forward_propagate", task_data, self.forward_propagated))
        self.statistics.current_record.queued(deserialization_latency)

    def forward_propagated(self, result):
        serialization_started = time()
        self.send_payload({ "pred": self.model_repository.transferToHost(result["pred"]) })
        serialization_latency = time() - serialization_started

        self.statistics.current_record.set_task_latency(result["latency"])
        self.statistics.current_record.set_serialization_latency(serialization_latency)

    def payload_received(self, payload):
        if "op" not in payload.keys():
            self.logger.error(f"Received unknown payload {payload}")
            return

        self.statistics.create_record(payload["op"])

        if payload["op"] == "initialize_stream":
            self.initialize_stream(payload)
        else:
            self.offload_task(payload)

    def connection_made(self, transport):
        self.statistics.connected()

        super().connection_made(transport)

    def connection_lost(self, exc):
        self.logger.info(self.statistics)

        super().connection_lost(exc)

    def send_payload(self, payload):
        super().send_payload(payload)

        self.statistics.task_completed()
