import asyncio

from .networking import TCPServer
from .task_executor import TaskExecutor

class RXPipe(TCPServer):
    """Class that handles the queuing of offloaded tasks data, and passes them to the application-specific task
    executor instance.

    User is not expected to implement a custom class in normal use cases, but instead use implementations provided by
    the framework. Currently only one such implementation exists, namely this class.
    """

    def __init__(self, benchmark_log_file_prefix = 'benchmark_sparse', benchmark_timeout = 60, **args):
        super().__init__(**args)

        self.task_executor = None

        self.benchmark_log_file_prefix = benchmark_log_file_prefix
        self.benchmark_timeout = benchmark_timeout
        self.warmed_up = False
        self.node = None

    def set_node(self, node):
        self.node = node
        self.logger = node.logger
        self.listen_address = node.config_manager.listen_address
        self.listen_port = node.config_manager.listen_port
        self.task_executor = node.task_executor

    async def handle_request(self, input_data : dict, context : dict) -> dict:
        output_data = await self.task_executor.execute_task(input_data)
        return output_data, context

    def request_processed(self, request_context : dict, processing_time : float):
        if self.node.monitor_client is not None:
            if self.warmed_up and (processing_time < self.benchmark_timeout):
                self.node.monitor_client.task_completed(processing_time)
            else:
                self.node.monitor_client.start_benchmark(f"{self.benchmark_log_file_prefix}-monitor")
                self.node.monitor_client.start_benchmark(f"{self.benchmark_log_file_prefix}-tasks", benchmark_type="ClientBenchmark")
                self.warmed_up = True
        self.logger.info(f"Processed offloaded task in {processing_time} seconds.")
