"""This module includes functionality related to stream api.
"""
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
        self.operators = set()
        self.streams = set()

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

    def connect_to_operator(self, operator : StreamOperator, output_stream):
        """Connects a stream to operator with given output stream.
        """
        self.operators.add((operator, output_stream))
        self.logger.info("Stream %s connected to operator %s with output stream %s", self, operator.name, output_stream)

    def connect_to_stream(self, stream):
        self.streams.add(stream)
        self.logger.info("Connected stream %s to stream %s", self, stream)

    def emit(self, data_tuple):
        """Sends a new data tuple to the connected operators and subscribed connections.
        """
        for operator, output_stream in self.operators:
            operator.buffer_input(data_tuple, output_stream.emit)

        for protocol in self.protocols:
            protocol.send_data_tuple(self.stream_alias or self.stream_id, data_tuple)

        for stream in self.streams:
            stream.emit(data_tuple)
