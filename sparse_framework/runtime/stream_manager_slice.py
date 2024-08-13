import asyncio
from graphlib import TopologicalSorter

from ..node import SparseSlice
from ..deploy.module_migrator_slice import SparseModuleMigratorSlice, SparseApp, UpstreamNode
from ..protocols import SparseProtocol
from ..stream_api import SparseStream

from .runtime_slice import SparseStreamRuntimeSlice

class SparseStreamManagerSlice(SparseSlice):
    """Sparse Stream Manager Slice receives applications to be deployed in the cluster, and decides the placement of
    sources, operators and sinks in the cluster. It then ensures that each stream is routed to the appropriate
    listeners.
    """
    def __init__(self, runtime_slice : SparseStreamRuntimeSlice, module_slice : SparseModuleMigratorSlice, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.upstream_nodes = set()

        #self.stream_replicas = []

        self.runtime_slice = runtime_slice
        self.module_slice = module_slice

    def add_upstream_node(self, protocol):
        self.upstream_nodes.add(UpstreamNode(protocol))

        self.logger.info("Added a new upstream node from %s", protocol.transport.get_extra_info('peername')[0])

    def remove_upstream_node(self, protocol):
        for node in self.upstream_nodes:
            if node.protocol == protocol:
                self.upstream_nodes.discard(node)
                self.logger.info("Removed upstream node from %s", protocol.transport.get_extra_info('peername')[0])
                return

    # def stream_received(self, stream_id, new_tuple, protocol = None):
    #     self.logger.info(f"Received stream replica {stream_id}")
    #     stream_replica = SparseStream(stream_id)

    #     if self.executor is not None and self.executor.operator is not None:
    #         self.output_stream = SparseStream()
    #         output_stream.add_protocol(protocol)
    #         stream_replica.add_executor(self.executor, output_stream)
    #         stream_replica.add_protocol(protocol)
    #     if self.sink is not None:
    #         stream_replica.add_sink(self.sink)

    #     self.stream_replicas.append(stream_replica)
    #     stream_replica.emit(new_tuple)

    # def tuple_received(self, stream_id, new_tuple, protocol = None):
    #     for stream in self.stream_replicas:
    #         if stream.stream_id == stream_id:
    #             stream.emit(new_tuple)
    #             return

    #     self.stream_received(stream_id, new_tuple, protocol)

    def deploy_node(self, node_name : str, destinations : set):
        """Deploys a Sparse application node to a cluster node from a local module.
        """
        factory, op_type, app = self.module_slice.get_factory(node_name)
        #self.logger.info("Deploying app node %s with destinations %s", node_name, destinations)

        if op_type == "Source":
            if self.config.root_server_address is None:
                for upstream_node in self.upstream_nodes:
                    connector_stream = self.runtime_slice.add_connector(upstream_node.protocol, destinations)
                    upstream_host = upstream_node.protocol.transport.get_extra_info('peername')[0]
                    app_dag = { node_name: { f"{upstream_host}:{connector_stream.stream_id}"} }
                    upstream_node.push_app(app, { "name": "stream_pace_steering", "dag": app_dag })
                    return

            updated_destinations = set()
            for destination in destinations:
                if ":" in destination:
                    [peer_ip, stream_id] = destination.split(":")
                    for upstream_node in self.upstream_nodes:
                        if peer_ip == upstream_node.protocol.transport.get_extra_info('peername')[0]:
                            self.logger.info("Creating a connector stream with id %s", stream_id)
                            connector_stream = SparseStream(stream_id)
                            connector_stream.add_protocol(upstream_node.protocol)
                            updated_destinations.add(connector_stream)
                else:
                    updated_dstinations.add(destination)

            self.runtime_slice.place_source(factory, updated_destinations)
        elif op_type == "Sink":
            self.runtime_slice.place_sink(factory)
            return
        elif op_type == "Operator":
            self.runtime_slice.place_operator(factory, destinations)
            return

    def deploy_app(self, app_dag : dict):
        """Deploys a Sparse application to a cluster.

        The application graph is sorted topologically so that each destination node is deployed before its sources.

        :param app_name: The name of the Sparse application to be deployed.
        :param app_dag: A dictionary representing the Directed Asyclic Graph of application nodes.
        """
        self.logger.info("Deploying app %s", app_dag)
        for node_name in TopologicalSorter(app_dag).static_order():
            if node_name in app_dag.keys():
                destinations = app_dag[node_name]
            else:
                destinations = {}


            self.deploy_node(node_name, destinations)
