import asyncio
from graphlib import TopologicalSorter

from ..node import SparseSlice
from .runtime_slice import SparseStreamRuntimeSlice
from ..deploy.module_migrator_slice import SparseModuleMigratorSlice, SparseApp, UpstreamNode
from ..protocols import SparseProtocol

class SparseStreamManagerSlice(SparseSlice):
    """Sparse Stream Manager Slice receives applications to be deployed in the cluster, and decides the placement of
    sources, operators and sinks in the cluster. It then ensures that each stream is routed to the appropriate
    listeners.
    """
    def __init__(self, runtime_slice : SparseStreamRuntimeSlice, module_slice : SparseModuleMigratorSlice, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sources = set()
        self.operators = set()
        self.sinks = set()

        self.upstream_nodes = set()

        self.stream_replicas = []

        self.runtime_slice = runtime_slice
        self.module_slice = module_slice

    def place_operator(self, operator_factory, destinations):
        operator = operator_factory()

        self.runtime_slice.add_operator(operator)

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

    def add_upstream_node(self, protocol):
        self.upstream_nodes.add(UpstreamNode(protocol))

        self.logger.info("Added a new upstream node from %s", protocol.transport.get_extra_info('peername')[0])

    def remove_upstream_node(self, protocol):
        for node in self.upstream_nodes:
            if node.protocol == protocol:
                self.upstream_nodes.discard(node)
                self.logger.info("Removed upstream node from %s", protocol.transport.get_extra_info('peername')[0])
                return

    def stream_received(self, stream_id, new_tuple, protocol = None):
        self.logger.info(f"Received stream replica {stream_id}")
        stream_replica = SparseStream(stream_id)

        if self.executor is not None and self.executor.operator is not None:
            self.output_stream = SparseStream()
            output_stream.add_protocol(protocol)
            stream_replica.add_executor(self.executor, output_stream)
            stream_replica.add_protocol(protocol)
        if self.sink is not None:
            stream_replica.add_sink(self.sink)

        self.stream_replicas.append(stream_replica)
        stream_replica.emit(new_tuple)

    def tuple_received(self, stream_id, new_tuple, protocol = None):
        for stream in self.stream_replicas:
            if stream.stream_id == stream_id:
                stream.emit(new_tuple)
                return

        self.stream_received(stream_id, new_tuple, protocol)

    def deploy_node(self, app_name : str, node_name : str, destinations : set):
        """Deploys a Sparse application node to a cluster node from a local module.
        """
        app = self.module_slice.get_app(app_name)
        app_module = app.load(self.config.app_repo_path)
        for source_factory in app_module.get_sources():
            if source_factory.__name__ == node_name:
                for upstream_node in self.upstream_nodes:
                    upstream_node.push_app(app)
                    return
                self.place_source(source_factory, destinations)
                return
        for sink_factory in app_module.get_sinks():
            if sink_factory.__name__ == node_name:
                self.place_sink(sink_factory)
                return
        for operator_factory in app_module.get_operators():
            if operator_factory.__name__ == node_name:
                self.place_operator(operator_factory, destinations)
                return

    def deploy_app(self, app_name : str, app_dag : dict):
        """Deploys a Sparse application to a cluster.

        The application graph is sorted topologically so that each destination node is deployed before its sources.

        :param app_name: The name of the Sparse application to be deployed.
        :param app_dag: A dictionary representing the Directed Asyclic Graph of application nodes.
        """
        for node_name in TopologicalSorter(app_dag).static_order():
            if node_name in app_dag.keys():
                destinations = app_dag[node_name]
            else:
                destinations = {}
            self.deploy_node(app_name, node_name, destinations)
