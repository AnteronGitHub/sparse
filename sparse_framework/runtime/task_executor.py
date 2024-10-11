import asyncio
import functools
import logging

from concurrent.futures import ThreadPoolExecutor
from time import time

from .operator import StreamOperator

class InputBufferedForOperatorNotFoundInRuntimeError(Exception):
    """Raised a module including a referenced operator cannot be found."""
    def __init__(self, operator_id : str):
        self.operator_id = operator_id

    def __str__(self):
        return f"Input buffered for operator {self.operator_id} that is not found in the runtime"

class TaskDispatchedForOperatorNotFoundInRuntimeError(Exception):
    """Raised a module including a referenced operator cannot be found."""
    def __init__(self, operator_id : str):
        self.operator_id = operator_id

    def __str__(self):
        return f"Dispatched a task for operator {self.operator_id} that is not found in the runtime"

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

    def add_operator(self, operator : StreamOperator):
        self.operators.add(operator)

    def get_operator(self, operator_id):
        for o in self.operators:
            if o.id == operator_id:
                return o
        return None

    async def start(self):
        loop = asyncio.get_running_loop()
        while True:
            operator_id = await self.queue.get()
            self.logger.debug("Dispatched tuple from queue (size %s)", self.queue.qsize())
            await loop.run_in_executor(self.executor, functools.partial(self.execute_task, operator_id))
            self.queue.task_done()

    def buffer_input(self, operator_id, input_data, result_callback):
        operator = self.get_operator(operator_id)
        if operator is None:
            raise InputBufferedForOperatorNotFoundInRuntimeError(operator_id)

        batch_index = operator.buffer_input(input_data, result_callback)

        if not operator.use_batching or batch_index == 0:
            self.logger.debug("Created task for operator %s", operator)
            self.queue.put_nowait(operator_id)

    def execute_task(self, operator_id):
        operator = self.get_operator(operator_id)
        if operator is None:
            raise TaskDispatchedForOperatorNotFoundInRuntimeError(operator_id)

        features, callbacks = operator.dispatch_buffer()

        pred = operator.call(features)

        operator.result_received(pred, callbacks)
