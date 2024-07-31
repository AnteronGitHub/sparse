import asyncio
from graphlib import TopologicalSorter
import importlib
import os
import pickle
import shutil

from ..node import SparseSlice
from ..runtime import SparseStreamManagerSlice
from ..protocols import SparseProtocol

class SparseApp:
    """A sparse app is a Python module that provides a set of sources, operators, and sinks.
    """
    def __init__(self, name : str, zip_path : str):
        self.name = name
        self.zip_path = zip_path
        self.app_module = None

    def load(self, app_repo_path : str):
        if self.app_module is None:
            shutil.unpack_archive(self.zip_path, os.path.join(app_repo_path, self.name))
            self.app_module = importlib.import_module(f".{self.name}", package="sparse_framework.apps")

        return self.app_module

class UpstreamNode:
    def __init__(self, protocol : SparseProtocol):
        self.protocol = protocol

    def push_app(self, app : SparseApp):
        self.protocol.migrate_app_module(app.zip_path)

class SparseModuleMigratorSlice(SparseSlice):
    """Sparse Module Migrator Slice migrates software modules for sources, operators and sinks over the network so that
    stream applications can be distributed in the cluster.
    """
    def __init__(self, stream_manager_slice : SparseStreamManagerSlice, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_manager_slice = stream_manager_slice
        self.upstream_nodes = set()
        self.apps = set()

    def get_futures(self, futures):
        futures.append(self.start_app_server())
        return futures

    async def start_app_server(self, listen_address = '0.0.0.0'):
        loop = asyncio.get_running_loop()

        server = await loop.create_server(lambda: SparseProtocol(self), \
                                          listen_address, \
                                          self.config.root_server_port)
        self.logger.info(f"Management plane listening to '{listen_address}:{self.config.root_server_port}'")
        async with server:
            await server.serve_forever()

    def add_upstream_node(self, protocol):
        self.upstream_nodes.add(UpstreamNode(protocol))

        self.logger.info("Added a new upstream node from %s", protocol.transport.get_extra_info('peername')[0])

    def remove_upstream_node(self, protocol):
        for node in self.upstream_nodes:
            if node.protocol == protocol:
                self.upstream_nodes.discard(node)
                self.logger.info("Removed upstream node from %s", protocol.transport.get_extra_info('peername')[0])
                return

    def add_app_module(self, name : str, zip_path : str):
        self.apps.add(SparseApp(name, zip_path))

    def get_app(self, name : str):
        for app in self.apps:
            if app.name == name:
                return app
        return None

    def deploy_node(self, app_name : str, node_name : str, destinations : set):
        """Deploys a Sparse application node to a cluster node from a local module.
        """
        app = self.get_app(app_name)
        app_module = app.load(self.config.app_repo_path)
        for source_factory in app_module.get_sources():
            if source_factory.__name__ == node_name:
                for upstream_node in self.upstream_nodes:
                    upstream_node.push_app(app)
                    return
                self.stream_manager_slice.place_source(source_factory, destinations)
                return
        for sink_factory in app_module.get_sinks():
            if sink_factory.__name__ == node_name:
                self.stream_manager_slice.place_sink(sink_factory)
                return
        for operator_factory in app_module.get_operators():
            if operator_factory.__name__ == node_name:
                self.stream_manager_slice.place_operator(operator_factory, destinations)
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

