import asyncio
import functools
import logging

from concurrent.futures import ThreadPoolExecutor

class TaskExecutor:
    """Common base class for task execution logic. This class implements potentially hardware-accelerated computations
    that are offloaded into worker nodes.

    User is expected to implement the computation logic by defining a custom execute_task() function. Additionally it
    is possible to implement custom initialization code by overriding optional start() hook.
    """
    def __init__(self, lock, memory_buffer, queue):
        self.logger = logging.getLogger("sparse")
        self.executor = ThreadPoolExecutor()
        self.memory_buffer = memory_buffer
        self.queue = queue
        self.lock = lock

    async def start(self):
        loop = asyncio.get_running_loop()
        while True:
            task_type, input_data, callback = await self.queue.get()
            await loop.run_in_executor(self.executor, functools.partial(self.execute_task, task_type, input_data, callback, self.lock))
            self.queue.task_done()

    def execute_task(self, input_data):
        raise "Task executor not implemented! See documentation on how to implement your own executor"
