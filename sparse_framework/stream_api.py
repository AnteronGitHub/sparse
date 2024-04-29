import asyncio
import uuid
import logging

__all__ = ["SparseSource", "SparseStream", "SparseSink", "SparseOperator"]

class SparseStream:
    def __init__(self, stream_id : str = None):
        self.logger = logging.getLogger("sparse")
        if stream_id is None:
            self.stream_id = str(uuid.uuid4())
        else:
            self.stream_id = stream_id

        self.protocol = None
        self.executor = None
        self.sink = None

        self.output_stream = None

    def add_protocol(self, protocol):
        self.protocol = protocol

    def add_executor(self, executor, output_stream):
        self.executor = executor
        self.output_stream = output_stream

    def add_sink(self, sink):
        self.sink = sink

    def emit(self, data_tuple):
        if self.sink is not None:
            self.sink.tuple_received(data_tuple)
        if self.executor is not None:
            self.executor.buffer_input(data_tuple, self.output_stream.emit, None)
        elif self.protocol is not None:
            payload = {'stream_id': self.stream_id, 'activation': data_tuple, "op": "offload_task"}
            self.protocol.send_payload(payload)

class SparseSource:
    def __init__(self, stream, no_samples = 64, target_latency = 200, use_scheduling = True):
        self.logger = logging.getLogger("sparse")
        self.id = str(uuid.uuid4())
        self.current_tuple = 0

        self.stream = stream
        self.target_latency = target_latency
        self.use_scheduling = use_scheduling

    def get_tuple(self):
        pass

    async def start(self):
        while True:
            self.stream.emit(self.get_tuple())
            self.current_tuple += 1
            await asyncio.sleep(self.target_latency / 1000.0)

class SparseSink:
    def __init__(self, logger):
        self.logger = logger

    def tuple_received(self, new_tuple):
        pass

class SparseOperator:
    def __init__(self, use_batching : bool = True):
        self.id = str(uuid.uuid4())
        self.batch_no = 0
        self.use_batching = use_batching

    def call(self, input_tuple):
        pass

