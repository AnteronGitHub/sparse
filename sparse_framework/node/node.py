import asyncio
import logging
import uuid

from .config_manager import ConfigManager
from ..stats.monitor_client import MonitorClient
from ..stats import MonitorDaemon

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
