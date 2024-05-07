import asyncio
import io
import logging
import pickle
import struct
import uuid

from sparse_framework.stats import RequestStatistics, ClientRequestStatistics, ServerRequestStatistics

class SparseProtocol(asyncio.Protocol):
    def __init__(self, stats_queue = None, request_statistics_factory = None):
        self.connection_id = str(uuid.uuid4())
        self.logger = logging.getLogger("sparse")
        self.payload_buffer = io.BytesIO()
        self.transport = None

        self.receiving_payload = False
        self.payload_size = 0

        if request_statistics_factory is None:
            self.request_statistics = None
        else:
            self.request_statistics = request_statistics_factory(self.connection_id, stats_queue)

        self.current_record = None

    def connection_made(self, transport):
        if self.request_statistics is not None:
            self.request_statistics.connected()

        peername = transport.get_extra_info('peername')
        self.transport = transport
        self.logger.debug(f"Connected to {peername}.")

    def send_payload(self, payload):
        payload_data = pickle.dumps(payload)
        payload_size = len(payload_data)

        self.transport.write(struct.pack("!Q", payload_size))
        self.transport.write(payload_data)

    def data_received(self, data):
        if self.receiving_payload:
            payload = data
        else:
            self.receiving_payload = True
            header = data[:8]
            self.payload_size = struct.unpack("!Q", header)[0]
            payload = data[8:]

        self.payload_buffer.write(payload)

        if self.payload_buffer.getbuffer().nbytes >= self.payload_size:
            payload_data = self.payload_buffer.getvalue()
            try:
                payload = pickle.loads(payload_data)
                self.payload_buffer = io.BytesIO()
                self.payload_received(payload)
                self.receiving_payload = False
            except pickle.UnpicklingError:
                self.logger.error(f"Deserialization error. {len(payload_data)} payload size, {self.payload_buffer.getbuffer().nbytes} buffer size.")


    def payload_received(self, payload):
        pass

class SparseClientProtocol(SparseProtocol):
    """Protocol for streaming data over a TCP connection.
    """
    def __init__(self, on_con_lost, node):
        super().__init__(stats_queue = node.stats_queue, request_statistics_factory = ClientRequestStatistics)

        self.on_con_lost = on_con_lost
        self.node = node

    def connection_made(self, transport):
        super().connection_made(transport)

        self.node.connected_to_server(self)

    def send_payload(self, payload):
        self.current_record = self.request_statistics.create_record("offload_task")
        self.current_record.processing_started()

        super().send_payload(payload)

        self.current_record.request_sent()

    def payload_received(self, payload):
        self.current_record.response_received()
        self.request_statistics.log_record(self.current_record)

        stream_id = payload['stream_id']
        self.logger.debug(f"Received payload for stream {stream_id}")

        if 'sync' in payload.keys():
            self.node.sync_received(self, payload['stream_id'], payload['sync'])

        self.node.tuple_received(payload['stream_id'], payload['pred'], protocol=self)

    def connection_lost(self, exc):
        self.logger.info(self.request_statistics)
        self.on_con_lost.set_result(self.request_statistics)

class SparseServerProtocol(SparseProtocol):
    def __init__(self, node, use_scheduling : bool = True):
        super().__init__(stats_queue = node.stats_queue, request_statistics_factory = ServerRequestStatistics)

        self.node = node

        self.use_scheduling = use_scheduling

    def payload_received(self, payload):
        if payload["type"] == "operator":
            self.logger.info("Received operator")
            return
        self.current_record = self.request_statistics.create_record(payload["op"])
        self.current_record.request_received()

        stream_id = payload['stream_id']
        new_tuple = payload['activation']
        self.node.tuple_received(stream_id, new_tuple, protocol=self)

    def send_payload(self, stream_id, result, batch_index = 0):
        payload = { "pred": result, 'stream_id': stream_id, 'sync': 0 }
        if self.use_scheduling:
            # Quantize queueing time to millisecond precision
            queueing_time_ms = int(self.request_statistics.get_queueing_time(self.current_record) * 1000)

            # Use externally measured median task latency
            task_latency_ms = 9

            # Use modulo arithmetics to spread batch requests
            sync_delay_ms = batch_index * task_latency_ms + queueing_time_ms % task_latency_ms

            self.current_record.set_sync_delay_ms(sync_delay_ms)
            payload["sync"] = sync_delay_ms / 1000.0

        super().send_payload(payload)

        self.current_record.response_sent()
        self.request_statistics.log_record(self.current_record)

    def connection_lost(self, exc):
        self.logger.info(self.request_statistics)

        super().connection_lost(exc)
