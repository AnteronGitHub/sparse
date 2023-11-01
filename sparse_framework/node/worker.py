import asyncio

from .node import Node
from .master import Master

from ..task_executor import TaskExecutor

class Worker(Node):
    def __init__(self, rx_protocol_factory, task_executor = TaskExecutor):
        Node.__init__(self)

        self.task_executor = task_executor
        self.rx_protocol_factory = rx_protocol_factory

        self.task_queue = None

    async def start_task_executor(self):
        await self.task_executor(self.task_queue).start()

    async def start_rx_pipe(self):
        loop = asyncio.get_running_loop()

        server = await loop.create_server(self.rx_protocol_factory(self.task_queue, self.stats_queue), \
                                          self.config_manager.listen_address, \
                                          self.config_manager.listen_port)
        async with server:
            await server.serve_forever()

    def get_futures(self):
        futures = super().get_futures()

        self.task_queue = asyncio.Queue()

        futures.append(self.start_task_executor())
        futures.append(self.start_rx_pipe())

        return futures

