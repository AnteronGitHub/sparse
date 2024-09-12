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
        self.operators = set()
        self.connector_streams = set()

    def get_futures(self, futures):
        self.executor = SparseTaskExecutor()

        futures.append(self.executor.start())

        return futures

    def add_connector(self, stream_type : str, protocol, destinations, stream_id : str = None):
        connector_stream = SparseStream(stream_type, stream_id)
        self.add_destinations(connector_stream, destinations)
        self.connector_streams.add(connector_stream)
        self.logger.info("Stream %s type '%s' listening to peer %s",
                         connector_stream.stream_id,
                         stream_type,
                         protocol.transport.get_extra_info('peername')[0])
        return connector_stream

    def add_destinations(self, stream : SparseStream, destinations : set):
        for o in self.operators:
            if o.name in destinations:
                stream.add_listener(o)
                self.logger.info("Connected stream %s to stream %s", o.output_stream.stream_id, stream.stream_id)

    def place_operator(self, operator_factory):
        """Places a stream operator to the local runtime.
        """
        o = operator_factory()
        self.executor.add_operator(o)
        self.operators.add(o)

        self.logger.info("Deployed '%s' operator with output stream %s", o.name, o.output_stream.stream_id)
        return o

    def find_operator(self, operator_name : str):
        for operator in self.operators:
            if operator.name == operator_name:
                return operator

        return None

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
