import asyncio
import logging
import os
import uuid

from dotenv import load_dotenv

from .stats import MonitorDaemon
from .task_executor import TaskExecutor

__all__ = ["Master",
           "Node",
           "Worker",
           "ConfigManager"]

class ConfigManager:
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
        self.model_server_address = os.environ.get('SPARSE_MODEL_SERVER_ADDRESS') or '127.0.0.1'
        self.model_server_port = os.environ.get('SPARSE_MODEL_SERVER_PORT') or 50006

class Node:
    def __init__(self, node_id : str = str(uuid.uuid4()), benchmark = True, log_level : int = logging.INFO):
        self.node_id = node_id

        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=log_level)
        self.logger = logging.getLogger("sparse")

        self.config_manager = ConfigManager()
        self.config_manager.load_config()

        self.stats_queue = None

    def get_futures(self):
        self.stats_queue = asyncio.Queue()
        self.monitor_daemon = MonitorDaemon(self.stats_queue)
        return [self.monitor_daemon.start()]

    async def start(self):
        await asyncio.gather(*self.get_futures())

class Master(Node):
    def __init__(self, tx_protocol_factory, **kwargs):
        Node.__init__(self, **kwargs)

        self.tx_protocol_factory = tx_protocol_factory

    async def start_tx_pipe(self, on_con_lost):
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()
        await loop.create_connection(self.tx_protocol_factory(on_con_lost, self.stats_queue),
                                     self.config_manager.upstream_host,
                                     self.config_manager.upstream_port)

    def get_futures(self):
        futures = super().get_futures()

        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()

        futures.append(self.start_tx_pipe(on_con_lost))
        futures.append(on_con_lost)

        return futures

class Worker(Node):
    def __init__(self, rx_protocol_factory, task_executor = TaskExecutor):
        Node.__init__(self)

        self.task_executor = task_executor
        self.rx_protocol_factory = rx_protocol_factory

        self.task_queue = None

    async def start_task_executor(self):
        await self.task_executor(self.task_queue).start()

    async def start_rx_pipe(self):
        loop = asyncio.get_running_loop()

        server = await loop.create_server(self.rx_protocol_factory(self.task_queue, self.stats_queue), \
                                          self.config_manager.listen_address, \
                                          self.config_manager.listen_port)
        async with server:
            await server.serve_forever()

    def get_futures(self):
        futures = super().get_futures()

        self.task_queue = asyncio.Queue()

        futures.append(self.start_task_executor())
        futures.append(self.start_rx_pipe())

        return futures

