import asyncio

from .node import Node
from .master import Master

from ..networking import TCPServer
from ..task_executor import TaskExecutor

class Worker(Node):
    def __init__(self, task_executor : TaskExecutor, rx_protocol_factory):
        Node.__init__(self)

        self.task_executor = task_executor
        self.rx_protocol_factory = rx_protocol_factory

        if isinstance(self, Master):
            self.task_executor.task_deployer = self.task_deployer

    def start(self):
        rx_pipe = TCPServer(self.config_manager.listen_address, self.config_manager.listen_port)
        asyncio.run(rx_pipe.serve(self.rx_protocol_factory))
