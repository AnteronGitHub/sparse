import asyncio
import uuid
import logging

from .protocols import SparseProtocol

__all__ = ["SparseStream", "SparseOperator"]

class SparseStream:
    def __init__(self, stream_id : str = None, stream_alias : str = None):
        self.logger = logging.getLogger("sparse")

        self.stream_id = str(uuid.uuid4()) if stream_id is None else stream_id
        self.stream_alias = stream_alias

        self.protocol = None
        self.operator = None

    def __str__(self):
        return self.stream_alias or self.stream_id

    def matches_selector(self, stream_selector : str) -> bool:
        return stream_selector == self.stream_alias \
                or stream_selector == self.stream_id

    def add_protocol(self, protocol : SparseProtocol):
        self.protocol = protocol
        self.logger.info("Stream id %s connected to peer %s",
                         self,
                         protocol.transport.get_extra_info('peername')[0])

    def add_operator(self, operator):
        self.operator = operator

    def emit(self, data_tuple):
        if self.operator is not None:
            self.operator.buffer_input(data_tuple)
        elif self.protocol is not None:
            self.protocol.send_data_tuple(self.stream_id, data_tuple)

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

