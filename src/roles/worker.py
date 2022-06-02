from ..node import Node
from ..config_manager import WorkerConfigManager

from .rx_pipe import get_supported_rx_pipe
from .task_executor import TaskExecutor

class Worker(Node):
    def __init__(self, task_executor : TaskExecutor, config_manager = WorkerConfigManager()):
        Node.__init__(self, config_manager = config_manager)
        self.task_executor = task_executor
        self.task_executor.set_logger(self.logger)
        self.rx_pipe = get_supported_rx_pipe(self.task_executor,
                                             self.config_manager.listen_address,
                                             self.config_manager.listen_port,
                                             legacy_asyncio = self.check_asyncio_use_legacy())
        self.rx_pipe.set_logger(self.logger)

    def start(self):
        self.task_executor.start()
        self.rx_pipe.start()
