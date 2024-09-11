import asyncio
import uuid
import logging

from .protocols import SparseProtocol, SourceProtocol

__all__ = ["SparseSource", "SparseStream", "SparseSink", "SparseOperator"]

class SparseStream:
    def __init__(self, stream_type : str, stream_id : str = None):
        self.logger = logging.getLogger("sparse")

        self.stream_type = stream_type
        if stream_id is None:
            self.stream_id = str(uuid.uuid4())
        else:
            self.stream_id = stream_id

        self.protocol = None
        self.operator = None
        self.sinks = set()

    def add_listener(self, listener):
        base_classes = [a.__name__ for a in [*listener.__class__.__bases__]]
        if "SparseSink" in base_classes:
            self.add_sink(listener)
        if "SparseOperator" in base_classes:
            self.add_operator(listener)

    def add_protocol(self, protocol : SparseProtocol):
        self.protocol = protocol
        self.logger.info("Stream id %s connected to peer %s",
                         self.stream_id,
                         protocol.transport.get_extra_info('peername')[0])

    def add_operator(self, operator):
        self.operator = operator

    def add_sink(self, sink):
        self.sinks.add(sink)

    def emit(self, data_tuple):
        for sink in self.sinks:
            sink.tuple_received(data_tuple)
        if self.operator is not None:
            self.operator.buffer_input(data_tuple)
        elif self.protocol is not None:
            self.protocol.send_data_tuple(self.stream_id, data_tuple)

class SparseSource:
    """Sparse source connects to a cluster and receives a stream id that it can use to send data to.
    """
    def __init__(self, no_samples = 64, target_latency = 200, use_scheduling = True):
        self.logger = logging.getLogger("sparse")
        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=logging.INFO)
        self.id = str(uuid.uuid4())
        self.current_tuple = 0

        self.stream = None

        self.target_latency = target_latency
        self.use_scheduling = use_scheduling

    @property
    def type(self):
        return self.__class__.__name__

    def get_tuple(self):
        pass

    async def connect(self, endpoint_host : str = "sparse-root", endpoint_port : int = 50006):
        """Called to start the data source. Connects the source to a cluster host and starts streaming data.
        """
        stream = await self.initialize_stream(endpoint_host, endpoint_port)
        self.stream = stream
        await self.start_stream()

    async def initialize_stream(self, endpoint_host : str, endpoint_port : int):
        loop = asyncio.get_running_loop()
        on_stream_initialized = loop.create_future()

        while True:
            try:
                self.logger.debug("Connecting to cluster endpoint on %s:%s.", endpoint_host, endpoint_port)
                await loop.create_connection(lambda: SourceProtocol(self.type, on_stream_initialized), \
                                             endpoint_host, \
                                             endpoint_port)
                stream = await on_stream_initialized
                return stream
            except ConnectionRefusedError:
                self.logger.warn("Connection refused. Re-trying in 5 seconds.")
                await asyncio.sleep(5)

    async def start_stream(self):
        while True:
            self.stream.emit(self.get_tuple())
            self.current_tuple += 1
            await asyncio.sleep(self.target_latency / 1000.0)

class SparseSink:
    def __init__(self, logger):
        self.logger = logger
        self.id = str(uuid.uuid4())

    @property
    def name(self):
        return self.__class__.__name__

    def tuple_received(self, new_tuple):
        pass

class SparseOperator:
    def __init__(self, use_batching : bool = True):
        self.id = str(uuid.uuid4())
        self.batch_no = 0
        self.use_batching = use_batching

        self.executor = None
        self.output_stream = SparseStream(self.name)

    @property
    def name(self):
        return self.__class__.__name__

    def set_executor(self, executor):
        self.executor = executor

    def buffer_input(self, data_tuple):
        self.executor.buffer_input(self.id, data_tuple, self.output_stream.emit, None)

    def call(self, input_tuple):
        pass

