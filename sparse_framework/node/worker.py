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
        queue = asyncio.Queue()
        rx_pipe = TCPServer(self.config_manager.listen_address, self.config_manager.listen_port)
        await asyncio.gather(self.task_executor.start(queue), rx_pipe.serve(lambda: self.rx_protocol(queue, self.task_executor, self.model_repository)))
