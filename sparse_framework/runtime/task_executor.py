import asyncio
import functools
import logging

from concurrent.futures import ThreadPoolExecutor
from time import time

from .io_buffer import SparsePytorchIOBuffer
from .operator import StreamOperator

class SparseTaskExecutor:
    """Common base class for task execution logic. This class implements potentially hardware-accelerated computations
    that are offloaded into worker nodes.

    User is expected to implement the computation logic by defining a custom execute_task() function. Additionally it
    is possible to implement custom initialization code by overriding optional start() hook.
    """
    def __init__(self):
        self.logger = logging.getLogger("SparseExecutor")
        self.executor = ThreadPoolExecutor()

        self.queue = asyncio.Queue()

        self.operators = set()
        self.memory_buffers = {}

    def add_operator(self, operator : StreamOperator):
        self.operators.add(operator)
        self.memory_buffers[operator.id] = SparsePytorchIOBuffer()

    def get_operator(self, operator_id):
        for o in self.operators:
            if o.id == operator_id:
                return o
        return None

    async def start(self):
        loop = asyncio.get_running_loop()
        while True:
            operator_id, callback = await self.queue.get()
            self.logger.debug("Dispatched tuple from queue (size %s)", self.queue.qsize())
            await loop.run_in_executor(self.executor, functools.partial(self.execute_task, operator_id, callback))
            self.queue.task_done()

    def buffer_input(self, operator_id, input_data, result_callback):
        memory_buffer = self.memory_buffers[operator_id]
        batch_index = memory_buffer.buffer_input(input_data, result_callback)

        operator = self.get_operator(operator_id)
        if operator is None:
            self.logger.error("Received input for an unregistered operator")
            return

        if not operator.use_batching or batch_index == 0:
            self.logger.debug("Buffered tuple for operator %s", operator)
            self.queue.put_nowait((operator_id, memory_buffer.result_received))

    def execute_task(self, operator_id, callback):
        operator = self.get_operator(operator_id)
        memory_buffer = self.memory_buffers[operator_id]
        if operator is None:
            self.logger.error("Dispatched task for an unregistered operator")
            return

        if operator.use_batching:
            features, callbacks = memory_buffer.dispatch_batch()
        else:
            features, callbacks = memory_buffer.pop_input()

        task_started_at = time()
        pred = operator.call(features)
        task_completed_at = time()

        # TODO: Log task completed timestamp
        #for record in statistics_records:
            # record.task_started(task_started_at, self.operator.batch_no)
            # record.task_completed(task_completed_at)
        operator.batch_no += 1

        callback(pred, callbacks, use_batching = operator.use_batching)
