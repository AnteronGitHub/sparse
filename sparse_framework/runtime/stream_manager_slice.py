import asyncio
from ..node import SparseSlice

from .runtime_slice import SparseStreamRuntimeSlice

class SparseStreamManagerSlice(SparseSlice):
    """Sparse Stream Manager Slice receives applications to be deployed in the cluster, and decides the placement of
    sources, operators and sinks in the cluster. It then ensures that each stream is routed to the appropriate
    listeners.
    """
    def __init__(self, runtime_slice : SparseStreamRuntimeSlice, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sources = set()
        self.operators = set()
        self.sinks = set()

        self.stream_replicas = []

        self.runtime_slice = runtime_slice

    def place_operator(self, operator_factory, destinations):
        operator = operator_factory()

        self.runtime_slice.add_operator(operator)

        for sink in self.sinks:
            if sink.name in destinations:
                operator.stream.add_listener(sink)

        for o in self.operators:
            if o.name in destinations:
                operator.stream.add_listener(o)

        self.operators.add(operator)
        self.logger.info(f"Placed operator '{operator.name}' with destinations {destinations}")

    def place_sink(self, sink_factory):
        sink = sink_factory(self.logger)
        self.sinks.add(sink)
        self.logger.info(f"Placed sink '{sink.name}'")

    def place_source(self, source_factory, destinations):
        source = source_factory()

        for operator in self.operators:
            if operator.name in destinations:
                source.stream.add_operator(operator)

        self.sources.add(source)

        loop = asyncio.get_running_loop()
        task = loop.create_task(source.start())

        self.logger.info(f"Placed source '{source.name}'")

    def stream_received(self, stream_id, new_tuple, protocol = None):
        self.logger.info(f"Received stream replica {stream_id}")
        stream_replica = SparseStream(stream_id)

        if self.executor is not None and self.executor.operator is not None:
            self.output_stream = SparseStream()
            output_stream.add_protocol(protocol)
            stream_replica.add_executor(self.executor, output_stream)
            stream_replica.add_protocol(protocol)
        if self.sink is not None:
            stream_replica.add_sink(self.sink)

        self.stream_replicas.append(stream_replica)
        stream_replica.emit(new_tuple)

    def tuple_received(self, stream_id, new_tuple, protocol = None):
        for stream in self.stream_replicas:
            if stream.stream_id == stream_id:
                stream.emit(new_tuple)
                return

        self.stream_received(stream_id, new_tuple, protocol)

