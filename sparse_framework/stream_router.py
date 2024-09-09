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

        self.cluster_connections = set()
        self.source_streams = set()

        self.runtime = runtime
        self.module_repo = module_repo

    def add_cluster_connection(self, protocol : SparseProtocol, direction : str):
        self.cluster_connections.add(ClusterConnection(protocol, direction))

        self.logger.info("Added %s connection with node %s", direction, protocol.transport.get_extra_info('peername')[0])

    def add_source_stream(self, stream_type : str, protocol : SparseProtocol):
        stream = self.runtime.add_connector(stream_type, protocol, {})
        self.source_streams.add(stream)

        return stream

    def remove_cluster_connection(self, protocol):
        for connection in self.cluster_connections:
            if connection.protocol == protocol:
                self.cluster_connections.discard(connection)
                self.logger.info("Removed %s connection with node %s", \
                                 connection.direction, \
                                 protocol.transport.get_extra_info('peername')[0])
                return

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

    def deploy_operator(self, source : SparseProtocol, operator_name : str, destinations : set):
        """Deploys a Sparse operator to a cluster node from a local module.
        """
        factory, op_type, module = self.module_repo.get_factory(operator_name)

        if op_type == "Source":
            if self.config.root_server_address is None:
                for connection in self.cluster_connections:
                    connector_stream = self.runtime.add_connector(connection.protocol, destinations)

                    upstream_host = connection.protocol.transport.get_extra_info('peername')[0]

                    app_dag = { operator_name: { f"{upstream_host}:{connector_stream.stream_id}"} }
                    connection.create_deployment({ "name": "stream_pace_steering", "dag": app_dag })
            else:
                self.runtime.place_source(factory, self.update_destinations(source, destinations))
        elif op_type == "Sink":
            self.runtime.place_sink(factory)
            return
        elif op_type == "Operator":
            self.runtime.place_operator(factory, destinations)
            return

    def distribute_module(self, source : SparseProtocol, module : SparseModule):
        for connection in self.cluster_connections:
            if connection.protocol != source:
                self.logger.info("Distributing module %s to node %s",
                                 module.name,
                                 connection.protocol.transport.get_extra_info('peername')[0])
                connection.transfer_module(module)

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
                    self.runtime.add_destinations(source_stream, destinations)
                    return

            self.deploy_operator(source, operator_name, destinations)
