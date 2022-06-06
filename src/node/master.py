from . import Node
from ..task_deployer import TaskDeployer, get_supported_task_deployer

class Master(Node):
    def __init__(self,
                 upstream_host : str = '127.0.0.1',
                 upstream_port : int = 50007,
                 task_deployer : TaskDeployer = None):
        Node.__init__(self)
        if task_deployer:
            self.task_deployer = task_deployer
        else:
            self.task_deployer = get_supported_task_deployer(upstream_host=self.config_manager.upstream_host,
                                                             upstream_port=self.config_manager.upstream_port,
                                                             legacy_asyncio=self.check_asyncio_use_legacy())

        self.task_deployer.logger = self.logger
        self.logger.info(f"Task deployer using upstream {self.task_deployer.upstream_host}:{self.task_deployer.upstream_port}")
