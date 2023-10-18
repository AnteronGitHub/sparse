import asyncio
import logging
import uuid
import sys

from .config_manager import ConfigManager
from ..stats.monitor_client import MonitorClient

class Node:
    def __init__(self, node_id : str = str(uuid.uuid4()), benchmark = True, log_level : int = logging.INFO):
        self.node_id = node_id

        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=log_level)
        self.logger = logging.getLogger("sparse")

        self.config_manager = ConfigManager()
        self.config_manager.load_config()

        if benchmark:
            self.logger.debug(f"Benchmarking execution")
            self.monitor_client = MonitorClient()
        else:
            self.logger.debug(f"Not benchmarking execution")
            self.monitor_client = None

    async def delay_coro(self, coro, *args, delay : float):
        await asyncio.sleep(delay)
        await coro(*args)

    def add_timeout(self, coro, *args, delay : float = 10):
        return asyncio.create_task(self.delay_coro(coro, *args, delay=delay))
