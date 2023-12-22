import asyncio

from torch.utils.data import DataLoader

from sparse_framework import SparseProtocol
from sparse_framework.stats import ServerRequestStatistics, ClientRequestStatistics

from .model_repository import DiskModelRepository

__all__ = ["InferenceClientProtocol", "InferenceServerProtocol", "ParameterClientProtocol", "ParameterServerProtocol"]

class ParameterClientProtocol(SparseProtocol):
    """Protocol for downloading model parameters over a TCP connection.
    """
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

class ParameterServerProtocol(SparseProtocol):
    """Protocol for serving model parameters over a TCP connection.
    """
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

class InferenceClientProtocol(SparseProtocol):
    """Protocol for serving models over a TCP connection.
    """
    def __init__(self,
                 data_source_id,
                 dataset,
                 model_meta_data,
                 on_con_lost,
                 no_samples,
                 use_scheduling,
                 target_latency,
                 stats_queue = None):
        super().__init__()
        self.dataloader = DataLoader(dataset, 1)
        self.model_meta_data = model_meta_data
        self.on_con_lost = on_con_lost
        self.no_samples = no_samples
        self.target_latency = target_latency
        self.use_scheduling = use_scheduling

        self.statistics = ClientRequestStatistics(self.connection_id, stats_queue)
        self.current_record = None

    def start_stream(self):
        self.current_record = self.statistics.create_record("initialize_stream")

        self.send_payload({ 'op': "initialize_stream", 'model_meta_data': self.model_meta_data })

        self.current_record.request_sent()

    def stream_started(self, result_data):
        self.current_record.response_received()
        self.statistics.log_record(self.current_record)
        offload_latency = self.statistics.get_offload_latency(self.current_record)
        self.logger.info(f"Initialized stream in {offload_latency:.2f} seconds with {1.0/self.target_latency:.2f} FPS target rate (Scheduling: {self.use_scheduling}).")

        # Start offloading tasks
        self.offload_task()

    def offload_task(self):
        self.current_record = self.statistics.create_record("offload_task")
        self.current_record.processing_started()
        self.no_samples -= 1
        features, labels = next(iter(self.dataloader))
        self.send_payload({ 'op': 'offload_task',
                            'activation': features,
                            'labels': labels,
                            'model_meta_data': self.model_meta_data })
        self.current_record.request_sent()

    def offload_task_completed(self, result_data):
        self.current_record.response_received()
        self.statistics.log_record(self.current_record)
        offload_latency = self.statistics.get_offload_latency(self.current_record)

        if (self.no_samples > 0):
            if self.use_scheduling and 'sync' in result_data.keys():
                sync = result_data['sync']
            else:
                sync = 0.0
            loop = asyncio.get_running_loop()
            loop.call_later(self.target_latency-offload_latency + sync if self.target_latency > offload_latency else 0, self.offload_task)
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

class InferenceServerProtocol(SparseProtocol):
    def __init__(self,
                 memory_buffer,
                 use_scheduling : bool,
                 use_batching : bool,
                 task_queue,
                 stats_queue,
                 lock):
        super().__init__()

        self.memory_buffer = memory_buffer
        self.task_queue = task_queue
        self.stats_queue = stats_queue

        self.use_scheduling = use_scheduling
        self.use_batching = use_batching

        self.statistics = ServerRequestStatistics(self.connection_id, stats_queue)
        self.lock = lock

        self.model_meta_data = None
        self.current_record = None

    def connection_made(self, transport):
        self.statistics.connected()

        super().connection_made(transport)

    def connection_lost(self, exc):
        self.logger.info(self.statistics)

        super().connection_lost(exc)

    def payload_received(self, payload):
        if "op" not in payload.keys():
            self.logger.error(f"Received unknown payload {payload}")
            return


        if payload["op"] == "initialize_stream":
            self.start_stream(payload)
        else:
            self.request_received(payload)

    def start_stream(self, payload):
        self.current_record = self.statistics.create_record(payload["op"])
        self.current_record.request_received()

        self.model_meta_data = payload['model_meta_data']
        self.memory_buffer.start_stream(self.model_meta_data, self.stream_started)

        self.current_record.task_queued()

    def stream_started(self, load_task):
        self.send_payload({ "statusCode": 200 })

        self.current_record.response_sent()
        self.statistics.log_record(self.current_record)

    def request_received(self, payload):
        self.current_record = self.statistics.create_record(payload["op"])
        self.current_record.request_received()

        index = self.memory_buffer.buffer_input(self.model_meta_data,
                                                payload['activation'],
                                                self.forward_propagated,
                                                self.current_record, self.lock)

        self.current_record.task_queued()
        if not self.use_batching or index == 0:
            self.task_queue.put_nowait(("forward_propagate", self.model_meta_data, self.memory_buffer.result_received))

    def forward_propagated(self, result, batch_index = 0):
        payload = { "pred": result }
        if self.use_scheduling:
            # Quantize queueing time to millisecond precision
            queueing_time_ms = int(self.statistics.get_queueing_time(self.current_record) * 1000)

            # Use externally measured median task latency
            task_latency_ms = 9

            # Use modulo arithmetics to spread batch requests
            sync_delay_ms = batch_index * task_latency_ms + queueing_time_ms % task_latency_ms

            self.current_record.set_sync_delay_ms(sync_delay_ms)
            payload["sync"] =  sync_delay_ms / 1000.0
        self.send_payload(payload)

        self.current_record.response_sent()
        self.statistics.log_record(self.current_record)
