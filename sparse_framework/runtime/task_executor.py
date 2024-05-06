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
    def __init__(self, lock, memory_buffer, queue):
        self.logger = logging.getLogger("sparse")
        self.executor = ThreadPoolExecutor()

        self.memory_buffer = memory_buffer
        self.queue = queue
        self.lock = lock

        self.operators = set()

    def add_operator(self, operator):
        operator.set_executor(self)
        self.operators.add(operator)

    def get_operator(self, operator_id):
        for o in self.operators:
            if o.id == operator_id:
                return o
        return None

    async def start(self):
        loop = asyncio.get_running_loop()
        while True:
            operator_id, callback = await self.queue.get()
            await loop.run_in_executor(self.executor, functools.partial(self.execute_task, operator_id, callback, self.lock))
            self.queue.task_done()

    def buffer_input(self, operator_id, input_data, result_callback, statistics_record):
        batch_index = self.memory_buffer.buffer_input(operator_id, input_data, result_callback, statistics_record, self.lock)
        #statistics_record.task_queued()

        operator = self.get_operator(operator_id)
        if operator is None:
            self.logger.error("Received input for an unregistered operator")
            return

        if not operator.use_batching or batch_index == 0:
            self.queue.put_nowait((operator_id, self.memory_buffer.result_received))

    def execute_task(self, operator_id, callback, lock):
        operator = self.get_operator(operator_id)
        if operator is None:
            self.logger.error("Dispatched task for an unregistered operator")
            return

        if operator.use_batching:
            features, callbacks, statistics_records = self.memory_buffer.dispatch_batch(operator_id, lock)
        else:
            features, callbacks, statistics_records = self.memory_buffer.pop_input(operator_id, lock)

        task_started_at = time()
        pred = operator.call(features)
        task_completed_at = time()

        #for record in statistics_records:
            # record.task_started(task_started_at, self.operator.batch_no)
            # record.task_completed(task_completed_at)
        operator.batch_no += 1

        callback(pred, callbacks, use_batching = operator.use_batching)
