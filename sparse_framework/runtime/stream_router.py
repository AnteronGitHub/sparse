import asyncio
from graphlib import TopologicalSorter

from ..node import SparseSlice
from ..deploy.module_repo import ModuleRepository, SparseApp, UpstreamNode
from ..protocols import SparseProtocol
from ..stream_api import SparseStream

from .runtime import SparseRuntime

class StreamRouter(SparseSlice):
    """Stream router then ensures that streams are routed according to application specifications. It receives
    applications to be deployed in the cluster, and decides the placement of sources, operators and sinks in the
    cluster.
    """
    def __init__(self, runtime : SparseRuntime, module_repo : ModuleRepository, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.upstream_nodes = set()

        self.runtime = runtime
        self.module_repo = module_repo

    def add_upstream_node(self, protocol, direction = 'ingress'):
        self.upstream_nodes.add(UpstreamNode(protocol))

        self.logger.info("Added %s connection with node %s", direction, protocol.transport.get_extra_info('peername')[0])

    def remove_upstream_node(self, protocol):
        for node in self.upstream_nodes:
            if node.protocol == protocol:
                self.upstream_nodes.discard(node)
                self.logger.info("Removed upstream node from %s", protocol.transport.get_extra_info('peername')[0])
                return

    def update_destinations(self, destinations):
        updated_destinations = set()
        for destination in destinations:
            if ":" in destination:
                [peer_ip, stream_id] = destination.split(":")
                for upstream_node in self.upstream_nodes:
                    if peer_ip == upstream_node.protocol.transport.get_extra_info('peername')[0]:
                        connector_stream = SparseStream(stream_id)
                        connector_stream.add_protocol(upstream_node.protocol)
                        updated_destinations.add(connector_stream)
            else:
                updated_destinations.add(destination)
        return updated_destinations

    def deploy_node(self, node_name : str, destinations : set):
        """Deploys a Sparse application node to a cluster node from a local module.
        """
        factory, op_type, app = self.module_repo.get_factory(node_name)

        if op_type == "Source":
            if self.config.root_server_address is None:
                for upstream_node in self.upstream_nodes:
                    connector_stream = self.runtime.add_connector(upstream_node.protocol, destinations)

                    upstream_host = upstream_node.protocol.transport.get_extra_info('peername')[0]
                    app_dag = { node_name: { f"{upstream_host}:{connector_stream.stream_id}"} }
                    upstream_node.push_app(app, { "name": "stream_pace_steering", "dag": app_dag })
            else:
                self.runtime.place_source(factory, self.update_destinations(destinations))
        elif op_type == "Sink":
            self.runtime.place_sink(factory)
            return
        elif op_type == "Operator":
            self.runtime.place_operator(factory, destinations)
            return

    def deploy_app(self, app_dag : dict):
        """Deploys a Sparse application to a cluster.

        The application graph is sorted topologically so that each destination node is deployed before its sources.

        :param app_name: The name of the Sparse application to be deployed.
        :param app_dag: A dictionary representing the Directed Asyclic Graph of application nodes.
        """
        self.logger.info("Creating deployment for app graph %s", app_dag)
        for node_name in TopologicalSorter(app_dag).static_order():
            if node_name in app_dag.keys():
                destinations = app_dag[node_name]
            else:
                destinations = {}


            self.deploy_node(node_name, destinations)
