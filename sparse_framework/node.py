import asyncio
from graphlib import TopologicalSorter
import importlib
import multiprocessing
import logging
import os
import pickle
import shutil
import uuid

from dotenv import load_dotenv

from .io_buffer import SparsePytorchIOBuffer
from .protocols import SparseClientProtocol, SparseServerProtocol, SparseAppReceiverProtocol, SparseAppDeployerProtocol
from .stats import MonitorDaemon
from .task_executor import SparseTaskExecutor
from .stream_api import SparseStream

__all__ = ["SparseNode", "SparseDeployer"]

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

class SparseNode:
    """Common base class for each Node in a Sparse cluster.

    Nodes maintain asynchronous task loop for distributed pipelines. Nodes add tasks such as opening or listening for
    network connections.
    """

    def __init__(self, node_id : str = str(uuid.uuid4()), log_level : int = logging.INFO):
        self.node_id = node_id

        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=log_level)
        self.logger = logging.getLogger("sparse")

        self.config = SparseNodeConfig()
        self.config.load_config()

        self.init_slices()

    def init_slices(self):
        runtime_slice = SparseStreamRuntimeSlice(self.config)
        stream_manager_slice = SparseStreamManagerSlice(runtime_slice, self.config)
        migrator_slice = SparseModuleMigratorSlice(stream_manager_slice, self.config)

        self.slices = [
                SparseQoSMonitorSlice(self.config),
                runtime_slice,
                stream_manager_slice,
                migrator_slice
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

class SparseSlice:
    def __init__(self, config):
        self.logger = logging.getLogger("sparse")
        self.config = config

    def get_futures(self, futures):
        return futures

class SparseQoSMonitorSlice(SparseSlice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats_queue = None

    def get_futures(self, futures):
        self.stats_queue = asyncio.Queue()
        monitor_daemon = MonitorDaemon(self.stats_queue)

        futures.append(monitor_daemon.start())
        return futures

class SparseMasterSlice(SparseSlice):
    def get_futures(self, futures):
        futures.append(self.connect_to_server(self.config.upstream_host, self.config.upstream_port))

        return futures

    async def connect_to_server(self, host, port):
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()

        while True:
            try:
                await loop.create_connection(lambda: SparseClientProtocol(on_con_lost, self), host, port)
                result = await on_con_lost
                return result
            except ConnectionRefusedError:
                self.logger.warn("Connection refused. Re-trying in 5 seconds.")
                await asyncio.sleep(5)

class SparseStreamRuntimeSlice(SparseSlice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = None
        self.io_buffer = None

    def get_futures(self, futures):
        m = multiprocessing.Manager()
        lock = m.Lock()

        task_queue = asyncio.Queue()

        self.io_buffer = SparsePytorchIOBuffer()
        self.executor = SparseTaskExecutor(lock, self.io_buffer, task_queue)

        futures.append(self.executor.start())
        futures.append(self.start_server(self.config.listen_address, self.config.listen_port))

        return futures

    def add_operator(self, operator):
        self.executor.add_operator(operator)

    def sync_received(self, protocol, stream_id, sync):
        self.logger.debug(f"Received {sync} s sync")
        if self.source is not None:
            if (self.source.no_samples > 0):
                offload_latency = protocol.request_statistics.get_offload_latency(protocol.current_record)

                if not self.source.use_scheduling:
                    sync = 0.0

                target_latency = self.source.target_latency

                loop = asyncio.get_running_loop()
                loop.call_later(target_latency-offload_latency + sync if target_latency > offload_latency else 0, self.source.emit)
            else:
                protocol.transport.close()

    async def start_server(self, addr, port):
        loop = asyncio.get_running_loop()

        self.logger.info(f"Data plane listening to '{addr}:{port}'")
        server = await loop.create_server(lambda: SparseServerProtocol(self), addr, port)
        async with server:
            await server.serve_forever()

class SparseStreamManagerSlice(SparseSlice):
    def __init__(self, runtime_slice : SparseStreamRuntimeSlice, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sources = set()
        self.operators = set()
        self.sinks = set()

        self.stream_replicas = []

        self.runtime_slice = runtime_slice

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

    def place_source(self, source_factory, destinations):
        source = source_factory()

        for operator in self.operators:
            if operator.name in destinations:
                source.stream.add_operator(operator)

        self.sources.add(source)

        loop = asyncio.get_running_loop()
        task = loop.create_task(source.start())

        self.logger.info(f"Placed source '{source.name}'")

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

class SparseModuleMigratorSlice(SparseSlice):
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

import os
import tempfile

class SparseDeployer(SparseNode):
    def __init__(self, app : dict, app_name = 'sparseapp', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = app
        self.app_name = app_name

    def archive_filepath(self):
        return os.path.join(tempfile.gettempdir(), f"{self.app_name}.zip")

    def archive_application(self, app_dir : str = '.'):
        self.logger.info("Archiving application")
        shutil.make_archive(os.path.join(tempfile.gettempdir(), self.app_name), 'zip', app_dir)

    def unpack_application(self, app_repo_dir : str = "applications"):
        shutil.unpack_archive(self.archive_filepath(), os.path.join(app_repo_dir, app_name))

    def connected(self, protocol):
        protocol.send_file(self.archive_filepath())

    def object_received(self, protocol, obj : dict):
        if obj["type"] == "ack":
            protocol.send_payload({"type": "app", "data": self.app})

    async def upload_to_server(self):
        self.archive_application()
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()

        while True:
            try:
                self.logger.info(f"Connecting to root server on sparse-worker:{self.config.root_server_port}.")
                await loop.create_connection(lambda: SparseAppDeployerProtocol(self, on_con_lost), \
                                             "sparse-worker", \
                                             self.config.root_server_port)
                await on_con_lost
                break
            except ConnectionRefusedError:
                self.logger.warn("Connection refused. Re-trying in 5 seconds.")
                await asyncio.sleep(5)

    def deploy(self):
        asyncio.run(self.upload_to_server())
