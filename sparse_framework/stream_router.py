import asyncio

from .module_repo import SparseModule
from .node import SparseSlice
from .protocols import SparseProtocol
from .runtime import SparseRuntime
from .stream_api import SparseStream

class SparseDeployment:
    """Sparse deployment specifies a data flow between sources, operators and sinks.
    """
    def __init__(self, name : str, dag : dict):
        self.name = name
        self.dag = dag

class ClusterConnection:
    """Cluster connection enables offloading operators to another cluster node.
    """
    def __init__(self, protocol : SparseProtocol, direction : str):
        self.protocol = protocol
        self.direction = direction

    def transfer_module(self, app : SparseModule):
        self.protocol.transfer_module(app)

    def create_deployment(self, app_dag : dict):
        self.protocol.create_deployment(app_dag)

class StreamRouter(SparseSlice):
    """Stream router then ensures that streams are routed according to application specifications. It receives
    applications to be deployed in the cluster, and decides the placement of sources, operators and sinks in the
    cluster.
    """
    def __init__(self, runtime : SparseRuntime, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.runtime = runtime

        self.cluster_connections = set()

        self.streams = set()

    def add_cluster_connection(self, protocol : SparseProtocol, direction : str):
        """Adds a connection to another cluster node for stream routing and operator migration.
        """
        cluster_connection = ClusterConnection(protocol, direction)
        self.cluster_connections.add(cluster_connection)
        self.logger.info("Added %s connection with node %s", direction, protocol)

        for connector_stream in self.streams:
            cluster_connection.protocol.send_create_connector_stream(connector_stream.stream_id,
                                                                     connector_stream.stream_alias)
            connector_stream.subscribe(cluster_connection.protocol)

    def remove_cluster_connection(self, protocol):
        """Removes a cluster connection.
        """
        for connection in self.cluster_connections:
            if connection.protocol == protocol:
                self.cluster_connections.discard(connection)
                self.logger.info("Removed %s connection with node %s", connection.direction, protocol)
                return

    def distribute_module(self, source : SparseProtocol, module : SparseModule):
        for connection in self.cluster_connections:
            if connection.protocol != source:
                self.logger.info("Distributing module %s to node %s", module.name, connection.protocol)
                connection.transfer_module(module)

    def create_connector_stream(self, \
                                source : SparseProtocol, \
                                stream_id : str = None, \
                                stream_alias : str = None):
        """Adds a new connector stream. A connector stream receives tuples over the network, either from another
        cluster node or a data source.
        """
        connector_stream = self.get_stream(stream_id, stream_alias)
        if source in connector_stream.protocols:
            connector_stream.protocols.remove(source)

        self.logger.info("Stream %s listening to source %s", connector_stream.stream_alias, source)

        # Broadcast to other cluster connections
        for connection in self.cluster_connections:
            if connection.protocol != source:
                self.logger.info("Broadcasting stream %s to peer %s", connector_stream, connection.protocol)
                connection.protocol.send_create_connector_stream(connector_stream.stream_id, connector_stream.stream_alias)
                connector_stream.subscribe(connection.protocol)

        return connector_stream

    def tuple_received(self, stream_selector : str, data_tuple):
        for stream in self.streams:
            if stream.matches_selector(stream_selector):
                stream.emit(data_tuple)
                self.logger.debug("Received data for stream %s", stream)
                return
        self.logger.warn("Received data for stream %s without a connector", stream_selector)

    def subscribe(self, stream_alias : str, protocol : SparseProtocol):
        """Subscribes a protocol to receive tuples in a data stream.
        """
        for stream in self.streams:
            if stream.matches_selector(stream_alias):
                stream.subscribe(protocol)
                return

        stream = self.get_stream(stream_alias=stream_alias)
        stream.subscribe(protocol)

    def connect_to_operators(self, stream : SparseStream, operator_names : set):
        """Adds destinations to a stream.
        """
        for o in self.runtime.operators:
            if o.name in operator_names:
                output_stream = self.get_stream(stream_alias=o.name)
                stream.connect_to_operator(o, output_stream)

    def get_stream(self, stream_id : str = None, stream_alias : str = None):
        """Returns a stream that matches the provided stream alias or stream id. If no stream exists, one is created.
        """
        stream_selector = stream_alias or stream_id
        for stream in self.streams:
            if stream.matches_selector(stream_selector):
                return stream

        stream = SparseStream(stream_id=stream_id, stream_alias=stream_alias)
        self.streams.add(stream)
        self.logger.info("Created stream %s", stream)

        return stream
