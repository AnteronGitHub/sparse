import asyncio
import functools
import logging

from concurrent.futures import ThreadPoolExecutor
from time import time

class SparseTaskExecutor:
    """Common base class for task execution logic. This class implements potentially hardware-accelerated computations
    that are offloaded into worker nodes.

    User is expected to implement the computation logic by defining a custom execute_task() function. Additionally it
    is possible to implement custom initialization code by overriding optional start() hook.
    """
    def __init__(self, lock, memory_buffer, queue, use_batching : bool = True):
        self.logger = logging.getLogger("sparse")
        self.executor = ThreadPoolExecutor()

        self.memory_buffer = memory_buffer
        self.queue = queue
        self.lock = lock

        self.use_batching = use_batching
        self.batch_no = 0
        self.operator = None

    def set_operator(self, operator):
        self.logger.info(f"Registered operator")
        self.operator = operator

    async def start(self):
        loop = asyncio.get_running_loop()
        while True:
            callback = await self.queue.get()
            await loop.run_in_executor(self.executor, functools.partial(self.execute_task, callback, self.lock))
            self.queue.task_done()

    def buffer_input(self, input_data, result_callback, statistics_record):
        batch_index = self.memory_buffer.buffer_input(input_data, result_callback, statistics_record, self.lock)
        statistics_record.task_queued()
        if not self.use_batching or batch_index == 0:
            self.queue.put_nowait((self.memory_buffer.result_received))

    def execute_task(self, callback, lock):
        if self.use_batching:
            features, callbacks, statistics_records = self.memory_buffer.dispatch_batch(lock)
        else:
            features, callbacks, statistics_records = self.memory_buffer.pop_input(lock)

        task_started_at = time()
        pred = self.operator.call(features)
        task_completed_at = time()

        for record in statistics_records:
            record.task_started(task_started_at, self.batch_no)
            record.task_completed(task_completed_at)
        self.batch_no += 1

        callback(pred, callbacks)
