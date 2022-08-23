import asyncio
import json
import logging
import socket

import time

from ..logging.file_logger import FileLogger

from .monitor.network_monitor import NetworkMonitor
from .monitor.time_monitor import TimeMonitor
from .monitor.training_monitor import TrainingMonitor

class MonitorServer():
    def __init__(self,
                 update_frequency_ps = 2,
                 timeout = 10,
                 socket_path = 'sparse-benchmark.socket',
                 log_file_prefix = 'executor-benchmark'):

        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=logging.INFO)
        self.logger = logging.getLogger("sparse")
        self.update_frequency_ps = update_frequency_ps
        self.timeout = timeout
        self.socket_path = socket_path
        self.log_file_prefix = log_file_prefix
        self.monitors = self._init_monitors()

        self.stats_logger = None
        self.previous_message = None

    def _init_monitors(self):
        monitors = []
        monitors.append(TimeMonitor())
        monitors.append(NetworkMonitor())
        monitors.append(TrainingMonitor())
        return monitors

    def _get_metrics(self):
        metrics = []
        for monitor in self.monitors:
            metrics += monitor.get_metrics()
        return metrics

    def _get_stats(self):
        stats = []
        for monitor in self.monitors:
            stats += monitor.get_stats()
        return stats

    def log_stats(self):
        if self.stats_logger is not None:
            self.stats_logger.log_row(self._get_stats())

    def start_benchmark(self):
        self.logger.info("Starting a new benchmark")
        self.stats_logger = FileLogger(file_prefix=f"{self.log_file_prefix}")
        self.stats_logger.log_row(self._get_metrics())

    def stop_benchmark(self):
        self.log_stats()
        self.stats_logger = None
        self.previous_message = None

    def batch_processed(self, batch_size):
        for monitor in self.monitors:
            if type(monitor).__name__ == 'TrainingMonitor':
                monitor.add_point(newly_processed_samples = batch_size)

    def task_processed(self):
        for monitor in self.monitors:
            if type(monitor).__name__ == 'TrainingMonitor':
                monitor.add_point(newly_processed_tasks = 1)

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
        payload = json.loads(input_data.decode())
        if payload['event'] == 'start':
            self.start_benchmark()
        elif payload['event'] == 'stop':
            self.stop_benchmark()
        elif payload['event'] == 'batch_processed':
            self.batch_processed(payload['batch_size'])
        elif payload['event'] == 'task_processed':
            self.task_processed()

        writer.write(input_data)
        writer.write_eof()
        writer.close()

    async def run_server(self):
        self.logger.info(f"Starting the monitoring server on '{self.socket_path}'")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server = await asyncio.start_unix_server(self.receive_message, path=self.socket_path)
        await server.serve_forever()

    async def run(self):
        await asyncio.gather(self.run_server(), self.run_monitor())

    def start(self):
        asyncio.run(self.run())

class MonitorClient():
    def __init__(self,
                 socket_path = 'sparse-benchmark.socket'):
        self.socket_path = socket_path

    async def _send_message(self, message):
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        writer.write(message)
        writer.write_eof()
        data = await reader.read(100)
        writer.close()

    async def start_benchmark(self):
        await self._send_message(json.dumps({"event": "start"}).encode())

    async def stop_benchmark(self):
        await self._send_message(json.dumps({"event": "stop"}).encode())

    async def batch_processed(self, batch_size : int):
        await self._send_message(json.dumps({"event": "batch_processed", "batch_size": batch_size}).encode())

    async def task_processed(self):
        await self._send_message(json.dumps({"event": "task_processed"}).encode())

