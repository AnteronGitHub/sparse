from utils import use_legacy_asyncio

from .rx_pipe import get_supported_rx_pipe
from .task_executor import TaskExecutor

class Worker:
    def __init__(self,
                 task_executor : TaskExecutor,
                 listen_address : str = '127.0.0.1',
                 listen_port : int = 50007):
        self.task_executor = task_executor
        self.rx_pipe = get_supported_rx_pipe(self.task_executor, listen_address, listen_port, legacy_asyncio = use_legacy_asyncio())

    def start(self):
        self.rx_pipe.start()
