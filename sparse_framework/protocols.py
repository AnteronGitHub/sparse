import asyncio
import io
import logging
import pickle
import struct
import uuid

from sparse_framework.stats import RequestStatistics, ClientRequestStatistics, ServerRequestStatistics

class SparseProtocol(asyncio.Protocol):
    """Common base class for all Sparse network protocols. Provides low-level implementations for sending byte files
    and Python objects.
    """
    def __init__(self):
        self.connection_id = str(uuid.uuid4())
        self.logger = logging.getLogger("sparse")
        self.transport = None

        self.data_buffer = io.BytesIO()
        self.receiving_data = False
        self.data_type = None
        self.data_size = 0

    def clear_buffer(self):
        self.data_buffer = io.BytesIO()
        self.receiving_data = False
        self.data_type = None
        self.data_size = 0

    def connection_made(self, transport):
        self.transport = transport
        peername = self.transport.get_extra_info('peername')
        self.logger.debug(f"Connected to {peername}.")

    def connection_lost(self, exc):
        peername = self.transport.get_extra_info('peername')
        self.logger.debug(f"{peername} disconnected.")

    def data_received(self, data : bytes):
        if self.receiving_data:
            payload = data
        else:
            self.receiving_data = True
            header = data[:9]
            [self.data_type, self.data_size] = struct.unpack("!sQ", header)
            payload = data[9:]

        self.data_buffer.write(payload)

        if self.data_buffer.getbuffer().nbytes >= self.data_size:
            self.message_received(self.data_type.decode(), self.data_buffer.getvalue())
            self.clear_buffer()

    def message_received(self, payload_type : str, data : bytes):
        if payload_type == "f":
            self.file_received(data)
        elif payload_type == "o":
            try:
                self.object_received(pickle.loads(data))
            except pickle.UnpicklingError:
                self.logger.error(f"Deserialization error. {len(data)} payload size, {self.payload_buffer.getbuffer().nbytes} buffer size.")

    def file_received(self, data : bytes):
        pass

    def object_received(self, obj : dict):
        pass

    def send_file(self, file_path):
        with open(file_path, "rb") as f:
            data_bytes = f.read()
            file_size = len(data_bytes)

            self.transport.write(struct.pack("!sQ", b"f", file_size))
            self.transport.write(data_bytes)

    def send_payload(self, payload : dict):
        payload_data = pickle.dumps(payload)
        payload_size = len(payload_data)

        self.transport.write(struct.pack("!sQ", b"o", payload_size))
        self.transport.write(payload_data)

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
