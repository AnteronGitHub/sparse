import asyncio
import logging
import os
import pickle
import uuid

from dotenv import load_dotenv

__all__ = ["SparseNode", "SparseDeployer", "SparseSlice"]

class SparseSlice:
    """Common super class for Sparse Node Slices.

    Slices are analogious to services in service oriented architecture. Each slice provides a coherent feature for a
    node. Additionally, slices may utilize features from other slices to provide higher-level features.
    """
    def __init__(self, config):
        self.logger = logging.getLogger("sparse")
        self.config = config

    def get_futures(self, futures):
        return futures

class SparseNodeConfig:
    def __init__(self):
        self.upstream_host = None
        self.upstream_port = None
        self.listen_address = None
        self.listen_port = None
        self.root_server_address = None
        self.root_server_port = None

    def load_config(self):
        load_dotenv(dotenv_path=".env")

        self.upstream_host = os.environ.get('MASTER_UPSTREAM_HOST') or '127.0.0.1'
        self.upstream_port = os.environ.get('MASTER_UPSTREAM_PORT') or 50007
        self.listen_address = os.environ.get('WORKER_LISTEN_ADDRESS') or '127.0.0.1'
        self.listen_port = os.environ.get('WORKER_LISTEN_PORT') or 50007
        self.root_server_address = os.environ.get('SPARSE_ROOT_SERVER_ADDRESS') or '0.0.0.0'
        self.root_server_port = os.environ.get('SPARSE_ROOT_SERVER_PORT') or 50006
        self.app_repo_path = os.environ.get('SPARSE_APP_REPO_PATH') or '/usr/lib/sparse_framework/apps'

from .deploy import SparseModuleMigratorSlice, SparseDeployer
from .runtime import SparseStreamRuntimeSlice, SparseStreamManagerSlice, SparseMasterSlice
from .stats import SparseQoSMonitorSlice

class SparseNode:
    """Common base class for each Node in a Sparse cluster.

    Nodes maintain the task loop for its components. Each functionality, including the runtime is implemented by slices
    that the node houses.
    """

    def __init__(self, node_id : str = str(uuid.uuid4()), log_level : int = logging.INFO):
        self.node_id = node_id

        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=log_level)
        self.logger = logging.getLogger("sparse")

        self.config = SparseNodeConfig()
        self.config.load_config()

        self.sparse_deployer = None

        self.init_slices()

    def init_slices(self):
        runtime_slice = SparseStreamRuntimeSlice(self.config)
        stream_manager_slice = SparseStreamManagerSlice(runtime_slice, self.config)
        migrator_slice = SparseModuleMigratorSlice(stream_manager_slice, self.config)
        self.sparse_deployer = SparseDeployer(self.config)

        self.slices = [
                SparseQoSMonitorSlice(self.config),
                runtime_slice,
                stream_manager_slice,
                migrator_slice,
                self.sparse_deployer
                ]

    def connected_to_server(self, protocol):
        if self.source is not None:
            self.source.stream.add_protocol(protocol)
            self.source.emit()

    def data_received(self, protocol):
        payload_type, data = protocol.data_type.decode(), protocol.data_buffer.getvalue()
        if payload_type == "f":
            self.file_received(protocol, data)
        elif payload_type == "o":
            try:
                self.object_received(protocol, pickle.loads(data))
            except pickle.UnpicklingError:
                self.logger.error(f"Deserialization error. {len(data)} payload size, {self.payload_buffer.getbuffer().nbytes} buffer size.")

    def get_futures(self, is_worker = True):
        """Collects node coroutines to be executed on startup.
        """
        futures = []

        for node_slice in self.slices:
            futures = node_slice.get_futures(futures)

        return futures

    async def start(self, is_root = True, operator_factory = None, source_factory = None, sink_factory = None):
        """Starts the main task loop by collecting all of the future objects.

        NB! When subclassing SparseNode instead of extending this function the user should use the get_futures
        function.
        """
        futures = self.get_futures(is_worker=is_root)

        await asyncio.gather(*futures)

    def deploy_app(self, app : dict):
        self.sparse_deployer.deploy(app)
