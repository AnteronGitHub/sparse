import asyncio
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

    def __init__(self, node_id : str = str(uuid.uuid4()), log_level : int = logging.INFO):
        self.node_id = node_id

        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=log_level)
        self.logger = logging.getLogger("sparse")

        self.config = SparseNodeConfig()
        self.config.load_config()

        self.stats_queue = None

    def get_futures(self):
        """Common base class for each Node in a Sparse cluster.

        Nodes maintain asynchronous task loop for distributed pipelines. Nodes add tasks such as opening or listening for
        network connections.
        """
        self.stats_queue = asyncio.Queue()
        self.monitor_daemon = MonitorDaemon(self.stats_queue)
        return [self.monitor_daemon.start()]

    async def start(self):
        """Starts the main task loop by collecting all of the future objects.

        NB! When subclassing SparseNode instead of extending this function the user should use the get_futures
        function.
        """
        await asyncio.gather(*self.get_futures())

    async def connect_to_server(self, protocol_factory, host, port, callback = None):
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()

        await loop.create_connection(protocol_factory(on_con_lost, self.stats_queue), host, port)
        result = await on_con_lost
        if callback is not None:
            callback(result)

        return result

    async def start_server(self, protocol_factory, addr, port):
        loop = asyncio.get_running_loop()

        self.logger.info(f"Listening to '{addr}:{port}'")
        server = await loop.create_server(protocol_factory, addr, port)
        async with server:
            await server.serve_forever()
