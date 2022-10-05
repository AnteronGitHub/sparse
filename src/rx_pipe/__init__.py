import asyncio
import logging
import time

from ..stats.monitor_client import MonitorClient
from ..task_executor import TaskExecutor

class RXPipe:
    """Class that handles the queuing of offloaded tasks data, and passes them to the application-specific task
    executor instance.

    User is not expected to implement a custom class in normal use cases, but instead use implementations provided by
    the framework. Currently only one such implementation exists, namely this class.
    """

    def __init__(self,
                 task_executor : TaskExecutor,
                 listen_address : str,
                 listen_port : int,
                 benchmark = True,
                 benchmark_log_file_prefix = 'benchmark_sparse',
                 benchmark_timeout = 30):
        self.listen_address = listen_address
        self.listen_port = listen_port
        self.task_executor = task_executor

        self.benchmark_log_file_prefix = benchmark_log_file_prefix
        self.benchmark_timeout = benchmark_timeout
        self.previous_message_received_at = None
        if benchmark:
            self.monitor_client = MonitorClient()
        else:
            self.monitor_client = None

    async def receive_task(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        """Generic callback function which passes offloaded task directly to the task executor.

        """
        if self.monitor_client is not None:
            if (self.previous_message_received_at is None) or (time.time() - self.previous_message_received_at >= self.benchmark_timeout):
                self.monitor_client.start_benchmark(self.benchmark_log_file_prefix)
            self.previous_message_received_at = time.time()

        self.logger.debug("Reading task input data...")
        input_data = await reader.read()

        self.logger.debug("Executing task...")
        result_data = await self.task_executor.execute_task(input_data)

        self.logger.debug("Responding with task output data...")
        writer.write(result_data)
        await writer.drain()
        writer.close()

        if self.monitor_client is not None:
            self.monitor_client.task_processed()
        self.logger.info("Finished streaming task result.")

    def set_logger(self, logger : logging.Logger):
        self.logger = logger

    def start(self):
        pass

def get_supported_rx_pipe(task_executor : TaskExecutor,
                          listen_address : str,
                          listen_port : int,
                          legacy_asyncio : bool = False,
                          benchmark_log_file_prefix = 'benchmark_sparse'):
    if legacy_asyncio:
        from .rx_pipe_legacy import RXPipeLegacy
        return RXPipeLegacy(task_executor, listen_address, listen_port, benchmark_log_file_prefix = benchmark_log_file_prefix)
    else:
        from .rx_pipe_latest import RXPipeLatest
        return RXPipeLatest(task_executor, listen_address, listen_port, benchmark_log_file_prefix = benchmark_log_file_prefix)
