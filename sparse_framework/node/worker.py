import asyncio

from .node import Node
from .master import Master

from ..networking import TCPServer
from ..task_executor import TaskExecutor

class Worker(Node):
    def __init__(self, rx_protocol_factory, task_executor = TaskExecutor):
        Node.__init__(self)

        self.task_executor = task_executor
        self.rx_protocol_factory = rx_protocol_factory

    def get_futures(self):
        futures = super().get_futures()

        task_queue = asyncio.Queue()
        task_executor = self.task_executor(task_queue)
        rx_pipe = TCPServer(self.config_manager.listen_address, self.config_manager.listen_port)

        futures.append(task_executor.start())
        futures.append(rx_pipe.serve(self.rx_protocol_factory(task_queue, self.stats_queue)))

        return futures

