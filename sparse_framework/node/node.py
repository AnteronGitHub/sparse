import logging
import sys

from .config_manager import ConfigManager
from ..stats.monitor_client import MonitorClient

class Node:
    def __init__(self, benchmark = True, log_level : int = logging.INFO):
        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=log_level)
        self.logger = logging.getLogger("sparse")
        self.config_manager = ConfigManager()
        self.config_manager.load_config()

        if benchmark:
            self.logger.info(f"Benchmarking execution")
            self.monitor_client = MonitorClient()
        else:
            self.logger.info(f"Not benchmarking execution")
            self.monitor_client = None
