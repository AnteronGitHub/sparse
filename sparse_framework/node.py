import asyncio
import multiprocessing
import logging
import os
import uuid

from dotenv import load_dotenv

from .io_buffer import SparsePytorchIOBuffer
from .protocols import SparseClientProtocol, SparseServerProtocol
from .stats import MonitorDaemon
from .task_executor import SparseTaskExecutor

__all__ = ["SparseNode"]

class SparseNodeConfig:
    def __init__(self):
        self.upstream_host = None
        self.upstream_port = None
        self.listen_address = None
        self.listen_port = None
        self.model_server_address = None
        self.model_server_port = None

    def load_config(self):
        load_dotenv(dotenv_path=".env")

        self.upstream_host = os.environ.get('MASTER_UPSTREAM_HOST') or '127.0.0.1'
        self.upstream_port = os.environ.get('MASTER_UPSTREAM_PORT') or 50007
        self.listen_address = os.environ.get('WORKER_LISTEN_ADDRESS') or '127.0.0.1'
        self.listen_port = os.environ.get('WORKER_LISTEN_PORT') or 50007
        self.model_server_address = os.environ.get('SPARSE_MODEL_SERVER_ADDRESS') or '0.0.0.0'
        self.model_server_port = os.environ.get('SPARSE_MODEL_SERVER_PORT') or 50006

class SparseNode:
    """Common base class for each Node in a Sparse cluster.

    Nodes maintain asynchronous task loop for distributed pipelines. Nodes add tasks such as opening or listening for
    network connections.
    """

    def __init__(self,
                 node_id : str = str(uuid.uuid4()),
                 log_level : int = logging.INFO,
                 operator_factory = None,
                 stream_factory = None,
                 sink_factory = None):
        self.node_id = node_id

        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=log_level)
        self.logger = logging.getLogger("sparse")

        self.config = SparseNodeConfig()
        self.config.load_config()

        self.executor = None
        self.io_buffer = None
        self.stats_queue = None

        self.operator_factory = operator_factory

        self.stream_factory = stream_factory
        self.sink_factory = sink_factory

        self.stream = None
        self.sink = None

    def connected_to_server(self, protocol):
        if self.stream_factory is not None:
            self.stream = self.stream_factory(protocol)

        self.stream.emit()

    def tuple_received(self, protocol, payload):
        if self.sink_factory is not None:
            if self.sink is None:
                self.sink = self.sink_factory(self.logger)
            self.sink.tuple_received(payload)

        if self.stream is not None:
            if (self.stream.no_samples > 0):
                offload_latency = protocol.request_statistics.get_offload_latency(protocol.current_record)

                if self.stream.use_scheduling:
                    sync = payload['sync']
                else:
                    sync = 0.0

                target_latency = self.stream.target_latency

                loop = asyncio.get_running_loop()
                loop.call_later(target_latency-offload_latency + sync if target_latency > offload_latency else 0, self.stream.emit)
            else:
                protocol.transport.close()

        if self.executor is not None:
            self.executor.buffer_input(payload["activation"], protocol.send_payload, protocol.current_record)

    def add_master_slice_futures(self, futures):
        if self.stream_factory is None or self.sink_factory is None:
            return futures

        futures.append(self.connect_to_server(self.config.upstream_host, self.config.upstream_port))

        return futures

    def add_worker_slice_futures(self, futures, operator_factory):
        if operator_factory is None:
            return futures

        m = multiprocessing.Manager()
        lock = m.Lock()

        task_queue = asyncio.Queue()

        self.io_buffer = SparsePytorchIOBuffer()
        self.executor = SparseTaskExecutor(lock, self.io_buffer, task_queue)
        self.executor.set_operator(operator_factory())

        futures.append(self.executor.start())
        futures.append(self.start_server(self.config.listen_address, self.config.listen_port))
        return futures

    def add_statistics_futures(self, futures):
        self.stats_queue = asyncio.Queue()
        monitor_daemon = MonitorDaemon(self.stats_queue)

        futures.append(monitor_daemon.start())
        return futures

    def get_futures(self):
        """Collects node coroutines to be executed on startup.
        """
        futures = []
        futures = self.add_statistics_futures(futures)
        futures = self.add_worker_slice_futures(futures, self.operator_factory)
        futures = self.add_master_slice_futures(futures)
        return futures

    async def start(self):
        """Starts the main task loop by collecting all of the future objects.

        NB! When subclassing SparseNode instead of extending this function the user should use the get_futures
        function.
        """
        await asyncio.gather(*self.get_futures())

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

    async def start_server(self, addr, port):
        loop = asyncio.get_running_loop()

        self.logger.info(f"Listening to '{addr}:{port}'")
        server = await loop.create_server(lambda: SparseServerProtocol(self), addr, port)
        async with server:
            await server.serve_forever()
