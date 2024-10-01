import asyncio
import uuid
import logging

from .protocols import SparseProtocol
from .runtime.operator import StreamOperator

__all__ = ["SparseStream"]

class SparseStream:
    def __init__(self, stream_id : str = None, stream_alias : str = None):
        self.logger = logging.getLogger("sparse")

        self.stream_id = str(uuid.uuid4()) if stream_id is None else stream_id
        self.stream_alias = stream_alias

        self.protocols = set()
        self.operator = None
        self.output_stream = None

    def __str__(self):
        return self.stream_alias or self.stream_id

    def matches_selector(self, stream_selector : str) -> bool:
        return stream_selector == self.stream_alias \
                or stream_selector == self.stream_id

    def subscribe(self, protocol : SparseProtocol):
        """Subscribes a protocol to receive stream tuples.
        """
        self.protocols.add(protocol)
        self.logger.info("Stream %s connected to peer %s", self, protocol)

    def add_operator(self, operator : StreamOperator, output_stream):
        self.operator = operator
        self.output_stream = output_stream
        self.logger.info("Stream %s connected to operator %s with output stream %s", self, operator.name, output_stream)

    def emit(self, data_tuple):
        if self.operator is not None:
            self.operator.buffer_input(data_tuple, self.output_stream.emit)
        for protocol in self.protocols:
            protocol.send_data_tuple(self.stream_alias or self.stream_id, data_tuple)

