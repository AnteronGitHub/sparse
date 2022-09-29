import asyncio
import json
import logging
import socket

import time

from .file_logger import FileLogger
from .monitor import MonitorContainer

class MonitorServer():
    def __init__(self,
                 update_frequency_ps = 2,
                 timeout = 30,
                 socket_path = '/data/sparse-benchmark.sock'):

        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=logging.INFO)
        self.logger = logging.getLogger("sparse")
        self.update_frequency_ps = update_frequency_ps
        self.timeout = timeout
        self.socket_path = socket_path
        self.monitor_container = MonitorContainer()

        self.stats_logger = None
        self.previous_message = None

    def log_stats(self):
        if self.stats_logger is not None:
            self.stats_logger.log_row(self.monitor_container.get_stats())

    def start_benchmark(self, log_file_prefix = 'benchmark_sparse'):
        self.logger.info("Starting a new benchmark")
        self.stats_logger = FileLogger(file_prefix=f"{log_file_prefix}")
        self.stats_logger.log_row(self.monitor_container.get_metrics())

    def stop_benchmark(self):
        self.log_stats()
        self.stats_logger = None
        self.previous_message = None

    async def run_monitor(self):
        self.logger.info("Starting monitor")
        while True:
            start_time = time.time()
            self.log_stats()
            time_elapsed = time.time() - start_time
            if self.previous_message is not None and start_time - self.previous_message >= self.timeout:
                self.logger.info("Stopping benchmark due to timeout")
                self.stop_benchmark()
            await asyncio.sleep(1.0/self.update_frequency_ps - time_elapsed)

    async def receive_message(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        """Generic callback function which passes offloaded task directly to the task executor.

        """
        self.previous_message = time.time()
        input_data = await reader.read()
        if len(input_data) == 0:
            self.logger.info("Received empty message")
            return

        writer.write("ACK".encode())
        writer.write_eof()
        writer.close()
        payload = json.loads(input_data.decode())

        if payload['event'] == 'start':
            self.start_benchmark(payload['log_file_prefix'])
        elif payload['event'] == 'stop':
            self.stop_benchmark()
        elif payload['event'] == 'batch_processed':
            self.monitor_container.batch_processed(payload['batch_size'])
        elif payload['event'] == 'task_processed':
            self.monitor_container.task_processed()

    async def run_server(self):
        self.logger.info(f"Starting the monitoring server on '{self.socket_path}'")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server = await asyncio.start_unix_server(self.receive_message, path=self.socket_path)
        await server.serve_forever()

    async def run(self):
        await asyncio.gather(self.run_server(), self.run_monitor())

    def start(self):
        asyncio.run(self.run())

