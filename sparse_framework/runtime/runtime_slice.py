import asyncio

from ..node import SparseSlice

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

    def get_futures(self, futures):
        self.executor = SparseTaskExecutor()

        futures.append(self.executor.start())

        return futures

    def place_operator(self, operator_factory, destinations):
        operator = operator_factory()

        self.executor.add_operator(operator)

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

    def place_source(self, source_factory, destinations : set):
        source = source_factory()

        for operator in self.operators:
            if operator.name in destinations:
                source.stream.add_operator(operator)

        for node in self.upstream_nodes:
            if node.protocol.transport.peername[0] in destinations:
                source.stream.add_protocol(node.protocol)

        self.sources.add(source)

        loop = asyncio.get_running_loop()
        task = loop.create_task(source.start())

        self.logger.info(f"Placed source '{source.name}'")

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
