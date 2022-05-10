from ..node import Node

from .rx_pipe import get_supported_rx_pipe
from .task_executor import TaskExecutor

class Worker(Node):
    def __init__(self,
                 task_executor : TaskExecutor,
                 listen_address : str = '127.0.0.1',
                 listen_port : int = 50007):
        super().__init__()
        self.task_executor = task_executor
        self.task_executor.set_logger(self.logger)
        self.rx_pipe = get_supported_rx_pipe(self.task_executor,
                                             listen_address,
                                             listen_port,
                                             legacy_asyncio = self.check_asyncio_use_legacy())
        self.rx_pipe.set_logger(self.logger)

    def start(self):
        self.task_executor.start()
        self.rx_pipe.start()
