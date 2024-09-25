import asyncio
from graphlib import TopologicalSorter

from .module_repo import ModuleRepository, SparseModule
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
    def __init__(self, runtime : SparseRuntime, module_repo : ModuleRepository, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.runtime = runtime
        self.module_repo = module_repo

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
            connector_stream.add_protocol(cluster_connection.protocol)

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
                connector_stream.add_protocol(connection.protocol)

        return connector_stream

    def tuple_received(self, stream_selector : str, data_tuple):
        for stream in self.streams:
            if stream.matches_selector(stream_selector):
                stream.emit(data_tuple)
                self.logger.debug("Received data for stream %s", stream)
                return
        self.logger.warn("Received data for stream %s without a connector", stream_selector)

    def subsribe_to_stream(self, stream_alias : str, protocol : SparseProtocol):
        for stream in self.streams:
            if stream.matches_selector(stream_alias):
                stream.add_protocol(protocol)
                return

        stream = self.get_stream(stream_alias=stream_alias)
        stream.add_protocol(protocol)

    def connect_to_operators(self, stream : SparseStream, operator_names : set):
        """Adds destinations to a stream.
        """
        for o in self.runtime.operators:
            if o.name in operator_names:
                output_stream = self.get_stream(stream_alias=o.name)
                stream.add_operator(o, output_stream)

    def deploy_operator(self, operator_name : str):
        """Deploys a Sparse operator to a cluster node from a local module.
        """
        self.logger.debug("Deploying operator '%s'", operator_name)
        operator_factory = self.module_repo.get_operator_factory(operator_name)

        if operator_factory is None:
            return None

        operator = self.runtime.place_operator(operator_factory)

        return operator

    def get_stream(self, stream_id : str = None, stream_alias : str = None):
        stream_selector = stream_alias or stream_id
        for stream in self.streams:
            if stream.matches_selector(stream_selector):
                return stream

        stream = SparseStream(stream_id=stream_id, stream_alias=stream_selector)
        self.streams.add(stream)
        self.logger.info("Created stream %s", stream)

        return stream

    def create_deployment(self, source : SparseProtocol, app_dag : dict):
        """Deploys a Sparse application to a cluster.

        The application graph is sorted topologically so that each destination node is deployed before its sources.

        :param app_name: The name of the Sparse application to be deployed.
        :param app_dag: A dictionary representing the Directed Asyclic Graph of application nodes.
        """
        self.logger.debug("Creating deployment for app graph %s", app_dag)

        # Place operators
        for stream_selector in TopologicalSorter(app_dag).static_order():
            self.deploy_operator(stream_selector)

        # Connect streams
        for stream_selector in TopologicalSorter(app_dag).static_order():
            destinations = app_dag[stream_selector] if stream_selector in app_dag.keys() else set()

            output_stream = self.get_stream(stream_alias=stream_selector)
            self.connect_to_operators(output_stream, destinations)
