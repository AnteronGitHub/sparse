import asyncio
import uuid
import logging

from ..protocols import SparseProtocol

class SourceProtocol(SparseProtocol):
    """Source protocol connects to a cluster end point and receives a stream id that can be used to transmit data
    tuples using the established connection.
    """
    def __init__(self, stream_type : str, on_stream_initialized : asyncio.Future):
        super().__init__()
        self.stream_type = stream_type
        self.on_stream_initialized = on_stream_initialized

    def connection_made(self, transport):
        super().connection_made(transport)
        self.create_source_stream(self.stream_type)

    def object_received(self, obj : dict):
        if obj["op"] == "create_source_stream":
            if obj["status"] == "success":
                stream_id = obj["stream_id"]
                from ..stream_api import SparseStream
                stream = SparseStream(self.stream_type, stream_id)
                stream.add_protocol(self)
                self.on_stream_initialized.set_result(stream)
        else:
            super().object_received(obj)

class SparseSource:
    """Implementation of a simulated Sparse cluster data source. A Sparse source connects to a cluster end point and
    receives a stream id that it can use to send data to.
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

