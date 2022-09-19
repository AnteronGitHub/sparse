import asyncio
import logging

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
                 benchmark_log_file_prefix = 'benchmark_sparse'):
        self.listen_address = listen_address
        self.listen_port = listen_port
        self.task_executor = task_executor
        self.benchmark = benchmark
        self.benchmark_log_file_prefix = benchmark_log_file_prefix

        self.monitor_client = False

    async def receive_task(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        """Generic callback function which passes offloaded task directly to the task executor.

        """
        if self.benchmark and not self.monitor_client:
            self.monitor_client = MonitorClient()
            self.monitor_client.start_benchmark(self.benchmark_log_file_prefix)

        self.logger.debug("Reading task input data...")
        input_data = await reader.read()

        self.logger.debug("Executing task...")
        result_data = await self.task_executor.execute_task(input_data)

        self.logger.debug("Responding with task output data...")
        writer.write(result_data)
        await writer.drain()
        writer.close()

        if self.benchmark and self.monitor_client:
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
