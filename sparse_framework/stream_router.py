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

        self.connector_streams = set()

        self.waiting_streams = {}

    def add_cluster_connection(self, protocol : SparseProtocol, direction : str):
        """Adds a connection to another cluster node for stream routing and operator migration.
        """
        cluster_connection = ClusterConnection(protocol, direction)
        self.cluster_connections.add(cluster_connection)

        for connector_stream in self.connector_streams:
            cluster_connection.protocol.send_create_connector_stream(connector_stream.stream_id,
                                                                     connector_stream.stream_alias)
            connector_stream.add_protocol(cluster_connection.protocol)

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

    def create_connector_stream(self, \
                                protocol : SparseProtocol, \
                                stream_id : str = None, \
                                stream_alias : str = None):
        """Adds a new connector stream. A connector stream receives tuples over the network, either from another
        cluster node or a data source.
        """
        connector_stream = SparseStream(stream_id=stream_id, stream_alias=stream_alias)
        self.connector_streams.add(connector_stream)

        self.logger.info("Created connector stream %s from source %s",
                         connector_stream.stream_alias,
                         protocol.transport.get_extra_info('peername')[0])

        # Check waiting streams
        for selector in [k for k in self.waiting_streams.keys() if k == stream_alias]:
            dest = self.waiting_streams[selector]

            if type(dest) == set:
                self.add_destinations(connector_stream, dest)
            else:
                connector_stream.add_protocol(dest)

            self.waiting_streams.pop(selector)

        # Broadcast to other cluster connections
        for connection in self.cluster_connections:
            if connection.protocol != protocol:
                connection.protocol.send_create_connector_stream(connector_stream.stream_id, connector_stream.stream_alias)
                connector_stream.add_protocol(connection.protocol)

        return connector_stream

    def tuple_received(self, stream_id : str, data_tuple):
        for stream in self.connector_streams:
            if stream.stream_id == stream_id:
                stream.emit(data_tuple)
                self.logger.debug("Received data for stream %s", stream_id)
                return
        self.logger.warn("Received data for stream %s without a connector", stream_id)

    def subsribe_to_stream(self, stream_alias : str, protocol : SparseProtocol):
        operator = self.runtime.find_operator(stream_alias)
        if operator is None:
            self.create_waiting_stream(stream_alias, protocol)
        else:
            self.logger.info("Subscribing to stream type '%s'", stream_alias)
            operator.output_stream.add_protocol(protocol)

    def create_waiting_stream(self, stream_selector : str, destination):
        self.waiting_streams[stream_selector] = destination
        self.logger.info("Waiting for stream '%s' to be available.", stream_selector)

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
                stream.add_operator(o)
                self.logger.info("Stream %s connected to stream %s", stream, o.output_stream)

    def deploy_operator(self, operator_name : str):
        """Deploys a Sparse operator to a cluster node from a local module.
        """
        self.logger.debug("Deploying operator '%s'", operator_name)
        operator_factory = self.module_repo.get_operator_factory(operator_name)

        if operator_factory is None:
            return None

        operator = self.runtime.place_operator(operator_factory)

        for operator_name in self.waiting_streams.keys():
            dst = self.waiting_streams[operator_name]
            if type(dst) == set:
                self.add_destinations(operator.output_stream, dst)
            else:
                operator.output_stream.add_protocol(dst)

        return operator

    def create_deployment(self, source : SparseProtocol, app_dag : dict):
        """Deploys a Sparse application to a cluster.

        The application graph is sorted topologically so that each destination node is deployed before its sources.

        :param app_name: The name of the Sparse application to be deployed.
        :param app_dag: A dictionary representing the Directed Asyclic Graph of application nodes.
        """
        self.logger.debug("Creating deployment for app graph %s", app_dag)

        for stream_selector in TopologicalSorter(app_dag).static_order():
            destinations = app_dag[stream_selector] if stream_selector in app_dag.keys() else set()

            for connector_stream in self.connector_streams:
                if connector_stream.matches_selector(stream_selector):
                    self.add_destinations(connector_stream, destinations)
                    return

            operator = self.deploy_operator(stream_selector)
            if operator is None:
                self.create_waiting_stream(stream_selector, destinations)
            else:
                self.add_destinations(operator.output_stream, destinations)
