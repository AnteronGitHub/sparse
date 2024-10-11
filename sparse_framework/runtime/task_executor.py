import asyncio
import functools
import logging

from concurrent.futures import ThreadPoolExecutor

from .operator import StreamOperator

class TaskExecutor:
    """Common base class for task execution logic. This class implements potentially hardware-accelerated computations
    that are offloaded into worker nodes.

    User is expected to implement the computation logic by defining a custom execute_task() function. Additionally it
    is possible to implement custom initialization code by overriding optional start() hook.
    """
    def __init__(self, task_queue : asyncio.Queue):
        self.logger = logging.getLogger("TaskExecutor")
        self.executor = ThreadPoolExecutor()

        self.queue = task_queue

    async def start(self):
        loop = asyncio.get_running_loop()
        while True:
            operator = await self.queue.get()
            self.logger.debug("Dispatched tuple from queue (size %s)", self.queue.qsize())
            await loop.run_in_executor(self.executor, functools.partial(self.execute_task, operator))
            self.queue.task_done()

    def execute_task(self, operator : StreamOperator):
        features, callbacks = operator.dispatch_buffer()

        pred = operator.call(features)

        operator.result_received(pred, callbacks)
