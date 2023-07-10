import asyncio
import logging
import pickle
import time

from ..task_executor import TaskExecutor

class RXPipeBase:
    """Class that handles the queuing of offloaded tasks data, and passes them to the application-specific task
    executor instance.

    User is not expected to implement a custom class in normal use cases, but instead use implementations provided by
    the framework. Currently only one such implementation exists, namely this class.
    """

    def __init__(self,
                 benchmark_log_file_prefix = 'benchmark_sparse',
                 benchmark_timeout = 30):
        self.listen_address = None
        self.listen_port = None
        self.task_executor = None

        self.benchmark_log_file_prefix = benchmark_log_file_prefix
        self.benchmark_timeout = benchmark_timeout
        self.previous_message_received_at = None
        self.warmed_up = False
        self.node = None

    def set_node(self, node):
        self.node = node
        self.logger = node.logger
        self.listen_address = node.config_manager.listen_address
        self.listen_port = node.config_manager.listen_port
        self.task_executor = node.task_executor

    def decode_request(self, payload : bytes) -> dict:
        return pickle.loads(payload)

    def encode_response(self, result_data : dict) -> bytes:
        return pickle.dumps(result_data)

    async def receive_task(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        """Generic callback function which passes offloaded task directly to the task executor.

        """
        self.previous_message_received_at = time.time()

        input_payload = await reader.read()

        result_data = await self.task_executor.execute_task(self.decode_request(input_payload))

        writer.write(self.encode_response(result_data))

        # TODO: Come up with a better workaround for IO errors. Ignore for now...
        try:
            await writer.drain()
        except BrokenPipeError:
            self.logger.info("Broken pipe during response stream. Ignoring...")
            pass
        writer.close()

        if self.node.monitor_client is not None:
            if self.warmed_up and (time.time() - self.previous_message_received_at < self.benchmark_timeout):
                self.node.monitor_client.task_processed()
            else:
                self.node.monitor_client.start_benchmark(self.benchmark_log_file_prefix)
                self.warmed_up = True
        self.logger.info("Finished streaming task result.")

    def start(self):
        pass
