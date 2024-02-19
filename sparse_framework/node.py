import asyncio
import multiprocessing
import logging
import os
import uuid

from dotenv import load_dotenv

from .stats import MonitorDaemon
from .task_executor import TaskExecutor

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
                 executor_factory = None,
                 server_protocol_factory = None):
        self.node_id = node_id

        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=log_level)
        self.logger = logging.getLogger("sparse")

        self.config = SparseNodeConfig()
        self.config.load_config()

        self.stats_queue = None

        self.executor_factory = executor_factory
        self.server_protocol_factory = server_protocol_factory

    def add_worker_slice_futures(self, futures, executor_factory, server_protocol_factory):
        if executor_factory is None or server_protocol_factory is None:
            return futures

        m = multiprocessing.Manager()
        lock = m.Lock()

        task_queue = asyncio.Queue()

        executor = self.executor_factory(lock, task_queue)

        futures.append(executor.start())
        futures.append(self.start_server(self.server_protocol_factory(executor, self.stats_queue), \
                                         self.config.listen_address, \
                                         self.config.listen_port))
        return futures

    def add_statistics_futures(self, futures):
        self.stats_queue = asyncio.Queue()
        monitor_daemon = MonitorDaemon(self.stats_queue)

        futures.append(monitor_daemon.start())
        return futures

    def get_futures(self):
        """Common base class for each Node in a Sparse cluster.

        Nodes maintain asynchronous task loop for distributed pipelines. Nodes add tasks such as opening or listening for
        network connections.
        """
        futures = []

        futures = self.add_statistics_futures(futures)
        futures = self.add_worker_slice_futures(futures, self.executor_factory, self.server_protocol_factory)
        return futures

    async def start(self):
        """Starts the main task loop by collecting all of the future objects.

        NB! When subclassing SparseNode instead of extending this function the user should use the get_futures
        function.
        """
        await asyncio.gather(*self.get_futures())

    async def connect_to_server(self, protocol_factory, host, port, callback = None):
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()

        while True:
            try:
                await loop.create_connection(protocol_factory(on_con_lost, self.stats_queue), host, port)
                result = await on_con_lost
                if callback is not None:
                    callback(result)

                return result
            except ConnectionRefusedError:
                self.logger.warn("Connection refused. Re-trying in 5 seconds.")
                await asyncio.sleep(5)

    async def start_server(self, protocol_factory, addr, port):
        loop = asyncio.get_running_loop()

        self.logger.info(f"Listening to '{addr}:{port}'")
        server = await loop.create_server(protocol_factory, addr, port)
        async with server:
            await server.serve_forever()
