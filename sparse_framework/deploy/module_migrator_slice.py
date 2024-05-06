import asyncio
from graphlib import TopologicalSorter
import importlib
import os
import pickle
import shutil

from ..node import SparseSlice
from ..protocols import SparseAppReceiverProtocol
from ..runtime import SparseStreamManagerSlice

class SparseModuleMigratorSlice(SparseSlice):
    """Sparse Module Migrator Slice migrates software modules for sources, operators and sinks over the network so that
    stream applications can be distributed in the cluster.
    """
    def __init__(self, stream_manager_slice : SparseStreamManagerSlice, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_manager_slice = stream_manager_slice

    def get_futures(self, futures):
        futures.append(self.start_app_server())
        return futures

    async def start_app_server(self):
        loop = asyncio.get_running_loop()

        server = await loop.create_server(lambda: SparseAppReceiverProtocol(self), \
                                          self.config.root_server_address, \
                                          self.config.root_server_port)
        self.logger.info(f"Management plane listening to '{self.config.root_server_address}:{self.config.root_server_port}'")
        async with server:
            await server.serve_forever()

    def file_received(self, protocol, data : bytes):
        app_archive_path = "/tmp/app.zip"
        with open(app_archive_path, "wb") as f:
            f.write(data)
        self.app_module_received(protocol, app_archive_path)
        protocol.send_payload({"type": "ack"})

    def deploy_node(self, node_name : str, destinations : set, module_name : str = "sparseapp"):
        app_module = importlib.import_module(f".{module_name}", package="sparse_framework.apps")
        for source_factory in app_module.get_sources():
            if source_factory.__name__ == node_name:
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
        for node_name in TopologicalSorter(app_dag).static_order():
            if node_name in app_dag.keys():
                destinations = app_dag[node_name]
            else:
                destinations = {}
            self.deploy_node(node_name, destinations)

    def app_received(self, protocol, app : dict):
        app_name = app["name"]
        app_dag = app["dag"]
        protocol.transport.close()

        self.logger.info(f"Received app '{app_name}'")
        self.deploy_app(app_name, app_dag)

    def app_module_received(self, protocol, app_archive_path : str, app_name : str = 'sparseapp'):
        shutil.unpack_archive(app_archive_path, os.path.join(self.config.app_repo_path, app_name))
        module_path = f"apps.{app_name}"

    def object_received(self, protocol, obj : dict):
        if obj["type"] == "app":
            app = obj["data"]
            self.app_received(protocol, app)

    def data_received(self, protocol):
        payload_type, data = protocol.data_type.decode(), protocol.data_buffer.getvalue()
        if payload_type == "f":
            self.file_received(protocol, data)
        elif payload_type == "o":
            try:
                self.object_received(protocol, pickle.loads(data))
            except pickle.UnpicklingError:
                self.logger.error(f"Deserialization error. {len(data)} payload size, {self.payload_buffer.getbuffer().nbytes} buffer size.")

