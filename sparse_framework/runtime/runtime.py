import asyncio

from ..node import SparseSlice
from ..stream_api import SparseStream

from .task_executor import SparseTaskExecutor

class SparseRuntime(SparseSlice):
    """Sparse Runtime maintains task executor, and the associated memory manager, for executing stream
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

    def add_connector(self, stream_type : str, protocol, destinations):
        connector_stream = SparseStream(stream_type)
        self.add_destinations(connector_stream, destinations)
        self.connector_streams.add(connector_stream)
        self.logger.info("Stream %s type '%s' listening to peer %s",
                         connector_stream.stream_id,
                         stream_type,
                         protocol.transport.get_extra_info('peername')[0])
        return connector_stream

    def add_destinations(self, stream : SparseStream, destinations : set):
        for sink in self.sinks:
            if sink.name in destinations:
                stream.add_listener(sink)
                self.logger.info("Connected sink %s to stream %s", sink.id, stream.stream_id)

        for o in self.operators:
            if o.name in destinations:
                stream.add_listener(o)
                self.logger.info("Connected stream %s to stream %s", o.output_stream.stream_id, stream.stream_id)

    def place_operator(self, operator_factory, destinations):
        o = operator_factory()
        self.executor.add_operator(o)
        self.operators.add(o)

        self.logger.info("Deployed '%s' operator with output stream %s", o.name, o.output_stream.stream_id)

        self.add_destinations(o.output_stream, destinations)

    def place_sink(self, sink_factory):
        sink = sink_factory(self.logger)
        self.sinks.add(sink)
        self.logger.info("Created sink %s for '%s'", sink.id, sink.name)

    def place_source(self, source_factory, destinations : set):
        source = source_factory()

        for destination in destinations:
            if type(destination) == SparseStream:
                source.stream = destination

        for operator in self.operators:
            if operator.name in destinations:
                source.stream.add_operator(operator)

        for protocol in self.connector_streams:
            if protocol.transport.get_extra_info('peername')[0] in destinations:
                source.stream.add_protocol(protocol)

        self.sources.add(source)

        loop = asyncio.get_running_loop()
        task = loop.create_task(source.start())

        self.logger.info(f"Created source for '%s' with output stream %s", source.type, source.stream.stream_id)

    def tuple_received(self, stream_id : str, data_tuple):
        for stream in self.connector_streams:
            if stream.stream_id == stream_id:
                stream.emit(data_tuple)
                self.logger.debug("Received data for stream %s", stream_id)
                return
        self.logger.warn("Received data for stream %s without a connector", stream_id)

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
