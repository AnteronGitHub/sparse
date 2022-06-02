import logging
import sys

from .config_manager import ConfigManager

class Node:
    def __init__(self, log_level : int = logging.INFO):
        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=log_level)
        self.logger = logging.getLogger("sparse")
        self.config_manager = ConfigManager()
        self.config_manager.load_config()

    def check_asyncio_use_legacy(self):
        if sys.version_info >= (3, 8, 10):
            self.logger.debug("Using latest asyncio implementation.")
            return False
        elif sys.version_info >= (3, 6, 9):
            self.logger.debug("Using legacy asyncio implementation.")
            return True
        else:
            self.logger.warning("The used Python interpreter is older than what is officially supported. This may " +
                                "cause some functionalities to break")
            return True
