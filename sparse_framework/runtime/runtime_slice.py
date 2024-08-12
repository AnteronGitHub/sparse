import asyncio

from ..node import SparseSlice
from ..stream_api import SparseStream

from .task_executor import SparseTaskExecutor

class SparseStreamRuntimeSlice(SparseSlice):
    """Sparse Stream Runtime Slice maintains task executor, and the associated memory manager, for executing stream
    application operations locally.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.executor = None
        self.sources = set()
        self.operators = set()
        self.sinks = set()
        self.connector_streams = set()

    def get_futures(self, futures):
        self.executor = SparseTaskExecutor()

        futures.append(self.executor.start())

        return futures

    def add_connector(self, protocol, destinations):
        connector_stream = SparseStream()
        self.add_destinations(connector_stream, destinations)
        self.connector_streams.add(connector_stream)

    def add_destinations(self, stream : str, destinations : set):
        for sink in self.sinks:
            if sink.name in destinations:
                stream.add_listener(sink)

        for o in self.operators:
            if o.name in destinations:
                stream.add_listener(o)

    def place_operator(self, operator_factory, destinations):
        operator = operator_factory()
        self.add_destinations(operator.stream, destinations)
        self.executor.add_operator(operator)
        self.operators.add(operator)

        self.logger.info(f"Placed operator '{operator.name}' with destinations {destinations}")

    def place_sink(self, sink_factory):
        sink = sink_factory(self.logger)
        self.sinks.add(sink)
        self.logger.info(f"Placed sink '{sink.name}'")

    def place_source(self, source_factory, destinations : set):
        source = source_factory()

        for operator in self.operators:
            if operator.name in destinations:
                source.stream.add_operator(operator)

        for protocol in self.connector_streams:
            if protocol.transport.peername[0] in destinations:
                source.stream.add_protocol(node.protocol)

        self.sources.add(source)

        loop = asyncio.get_running_loop()
        task = loop.create_task(source.start())

        self.logger.info(f"Placed source '{source.name}'")

    def tuple_received(self, stream_id : str, data_tuple):
        self.logger.info("Received data tuple for stream %s", stream_id)
        for stream in self.connector_streams:
            if stream.id == stream_id:
                stream.emit(data_tuple)
                return

    def sync_received(self, protocol, stream_id, sync):
        self.logger.debug(f"Received {sync} s sync")
        if self.source is not None:
            if (self.source.no_samples > 0):
                offload_latency = protocol.request_statistics.get_offload_latency(protocol.current_record)

                if not self.source.use_scheduling:
                    sync = 0.0

                target_latency = self.source.target_latency

                loop = asyncio.get_running_loop()
                loop.call_later(target_latency-offload_latency + sync if target_latency > offload_latency else 0, self.source.emit)
            else:
                protocol.transport.close()
