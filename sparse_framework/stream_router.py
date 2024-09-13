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

        self.source_streams = set()
        self.connector_streams = set()

    def add_cluster_connection(self, protocol : SparseProtocol, direction : str):
        """Adds a connection to another cluster node for stream routing and operator migration.
        """
        self.cluster_connections.add(ClusterConnection(protocol, direction))

        self.logger.info("Added %s connection with node %s", direction, protocol.transport.get_extra_info('peername')[0])

    def remove_cluster_connection(self, protocol):
        """Removes a cluster connection.
        """
        for connection in self.cluster_connections:
            if connection.protocol == protocol:
                self.cluster_connections.discard(connection)
                self.logger.info("Removed %s connection with node %s", \
                                 connection.direction, \
                                 protocol.transport.get_extra_info('peername')[0])
                return

    def distribute_module(self, source : SparseProtocol, module : SparseModule):
        for connection in self.cluster_connections:
            if connection.protocol != source:
                self.logger.info("Distributing module %s to node %s",
                                 module.name,
                                 connection.protocol.transport.get_extra_info('peername')[0])
                connection.transfer_module(module)

    def add_connector(self, stream_type : str, protocol, stream_id : str = None):
        """Adds a new connector stream. A connector stream receives tuples over the network, either from another
        cluster node or a data source.
        """
        connector_stream = SparseStream(stream_type, stream_id)
        self.connector_streams.add(connector_stream)
        self.logger.info("Stream %s type '%s' listening to peer %s",
                         connector_stream.stream_id,
                         stream_type,
                         protocol.transport.get_extra_info('peername')[0])
        return connector_stream

    def tuple_received(self, stream_id : str, data_tuple):
        for stream in self.connector_streams:
            if stream.stream_id == stream_id:
                stream.emit(data_tuple)
                self.logger.debug("Received data for stream %s", stream_id)
                return
        self.logger.warn("Received data for stream %s without a connector", stream_id)

    def add_source_stream(self, stream_type : str, protocol : SparseProtocol, stream_id : str = None):
        """Creates a connector for a source stream, and broadcasts tuples to other cluster nodes.
        """
        stream = self.add_connector(stream_type, protocol, stream_id)

        for connection in self.cluster_connections:
            if connection.protocol != protocol:
                connection.protocol.create_source_stream(stream_type, stream.stream_id)
                stream.add_protocol(connection.protocol)

        self.source_streams.add(stream)

        return stream

    def subsribe_to_stream(self, stream_type, protocol : SparseProtocol):
        operator = self.runtime.find_operator(stream_type)
        self.logger.info("Subscribing to stream type '%s'", stream_type)
        if operator is None:
            return False
        else:
            operator.output_stream.add_protocol(protocol)
            return True

    def update_destinations(self, source : SparseProtocol, destinations : set):
        source_ip = source.transport.get_extra_info('peername')[0]
        updated_destinations = set()
        for destination in destinations:
            if ":" in destination:
                [peer_ip, stream_id] = destination.split(":")
                for connection in self.cluster_connections:
                    if source_ip == connection.protocol.transport.get_extra_info('peername')[0]:
                        connector_stream = SparseStream(stream_id)
                        connector_stream.add_protocol(connection.protocol)
                        updated_destinations.add(connector_stream)
            else:
                updated_destinations.add(destination)
        return updated_destinations

    def add_destinations(self, stream : SparseStream, destinations : set):
        """Adds destinations to a stream.
        """
        for o in self.runtime.operators:
            if o.name in destinations:
                stream.add_listener(o)
                self.logger.info("Connected stream %s to stream %s", o.output_stream.stream_id, stream.stream_id)

    def deploy_operator(self, operator_name : str):
        """Deploys a Sparse operator to a cluster node from a local module.
        """
        self.logger.info("Deploying operator '%s'", operator_name)
        operator_factory = self.module_repo.get_operator_factory(operator_name)

        return self.runtime.place_operator(operator_factory)

    def create_deployment(self, source : SparseProtocol, app_dag : dict):
        """Deploys a Sparse application to a cluster.

        The application graph is sorted topologically so that each destination node is deployed before its sources.

        :param app_name: The name of the Sparse application to be deployed.
        :param app_dag: A dictionary representing the Directed Asyclic Graph of application nodes.
        """
        self.logger.info("Creating deployment for app graph %s", app_dag)

        for operator_name in TopologicalSorter(app_dag).static_order():
            if operator_name in app_dag.keys():
                destinations = app_dag[operator_name]
            else:
                destinations = {}

            for source_stream in self.source_streams:
                if source_stream.stream_type == operator_name:
                    self.add_destinations(source_stream, destinations)
                    return

            operator = self.deploy_operator(operator_name)
            self.add_destinations(operator.output_stream, destinations)
