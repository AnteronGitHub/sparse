import asyncio

from .node import Node
from .master import Master

from ..networking import TCPServer
from ..task_executor import TaskExecutor

class Worker(Node):
    def __init__(self, task_executor : TaskExecutor, rx_protocol):
        Node.__init__(self)

        self.task_executor = task_executor
        self.rx_protocol = rx_protocol

        if isinstance(self, Master):
            self.task_executor.task_deployer = self.task_deployer

    async def start(self):
        task_queue = asyncio.Queue()
        task_executor = self.task_executor(task_queue)
        rx_pipe = TCPServer(self.config_manager.listen_address, self.config_manager.listen_port)
        await asyncio.gather(task_executor.start(),
                             rx_pipe.serve(lambda: self.rx_protocol(task_queue,
                                                                    task_executor,
                                                                    self.model_repository)))
