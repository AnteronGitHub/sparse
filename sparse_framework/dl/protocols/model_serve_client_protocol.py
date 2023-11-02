import asyncio

from torch.utils.data import DataLoader

from sparse_framework.networking.protocols import SparseProtocol
from sparse_framework.stats import ClientRequestStatistics

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
