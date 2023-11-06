import asyncio
from time import time

from torch.utils.data import DataLoader

from sparse_framework import SparseProtocol
from sparse_framework.stats import ServerRequestStatistics, ClientRequestStatistics

from .model_repository import DiskModelRepository

__all__ = ["ModelServeClientProtocol", "ModelServeServerProtocol"]

TARGET_FPS = 5.0

class ModelDownloaderClientProtocol(SparseProtocol):
    def __init__(self, model_meta_data, on_con_lost):
        super().__init__()

        self.model_meta_data = model_meta_data
        self.on_con_lost = on_con_lost

    def download_model(self):
        self.send_payload({ 'method' : 'get_model', 'model_meta_data': self.model_meta_data })

    def model_downloaded(self, model):
        self.on_con_lost.set_result(model)

    def payload_received(self, payload):
        model = payload["model"]
        loss_fn = payload["loss_fn"]
        optimizer = payload["optimizer"]

        if "model" in payload.keys():
            self.model_downloaded(payload["model"])

    def connection_made(self, transport):
        super().connection_made(transport)

        self.download_model()

class ModelDownloaderServerProtocol(SparseProtocol):
    def __init__(self):
        super().__init__()

        self.model_repository = DiskModelRepository()

    def payload_received(self, payload):
        method = payload["method"]

        if method == "get_model":
            model_meta_data = payload["model_meta_data"]
            model, loss_fn, optimizer = self.model_repository.get_model(model_meta_data)
            self.send_payload({ "model": model, "loss_fn": loss_fn, "optimizer": optimizer })
        else:
            self.logger.error(f"Received request for unknown method '{method}'.")

class ModelServeClientProtocol(SparseProtocol):
    def __init__(self, data_source_id, dataset, model_meta_data, on_con_lost, no_samples, target_rate = 1/TARGET_FPS, stats_queue = None):
        super().__init__()
        self.dataloader = DataLoader(dataset, 1)
        self.model_meta_data = model_meta_data
        self.on_con_lost = on_con_lost
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

    def payload_received(self, payload):
        if "statusCode" in payload.keys():
            self.stream_started(payload)
        else:
            self.offload_task_completed(payload)

    def connection_made(self, transport):
        self.statistics.connected()

        super().connection_made(transport)

        self.start_stream()

    def connection_lost(self, exc):
        self.on_con_lost.set_result(self.statistics)

    def send_payload(self, payload):
        super().send_payload(payload)

        self.statistics.request_sent()

class ModelServeServerProtocol(SparseProtocol):
    def __init__(self, node, task_queue, stats_queue):
        super().__init__()

        self.task_queue = task_queue
        self.stats_queue = stats_queue
        self.memory_buffer = node.get_memory_buffer()
        self.statistics = ServerRequestStatistics(node.node_id, stats_queue)

        self.model_meta_data = None

    def start_stream(self, payload):
        self.model_meta_data = payload['model_meta_data']

        self.memory_buffer.start_stream(self.model_meta_data, self.stream_started)

        self.statistics.current_record.queued()

    def stream_started(self, load_task):
        self.send_payload({ "statusCode": 200 })

    def offload_task(self, payload):
        self.memory_buffer.input_received(self.model_meta_data,
                                          payload['activation'],
                                          self.task_queue,
                                          self.statistics,
                                          self.forward_propagated)

    def forward_propagated(self, result):
        self.send_payload({ "pred": result })

    def payload_received(self, payload):
        if "op" not in payload.keys():
            self.logger.error(f"Received unknown payload {payload}")
            return

        self.statistics.create_record(payload["op"])

        if payload["op"] == "initialize_stream":
            self.start_stream(payload)
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
