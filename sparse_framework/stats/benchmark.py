import time

from .file_logger import FileLogger
from .monitor import MonitorContainer

class Benchmark():
    def __init__(self, benchmark_id, log_file_prefix, stop_callback, timeout = 30):
        self.benchmark_id = benchmark_id
        self.stats_logger = FileLogger(file_prefix=f"{log_file_prefix}")
        self.stop_callback = stop_callback
        self.timeout = timeout

        self.monitor_container = MonitorContainer()
        self.stats_logger.log_row(self.monitor_container.get_metrics())

        self.previous_message_received_at = time.time()

    def log_stats(self):
        self.stats_logger.log_row(self.monitor_container.get_stats())

        if time.time() - self.previous_message_received_at >= self.timeout:
            self.stop_callback(self)

    def receive_message(self, payload):
        self.previous_message_received_at = time.time()
        if payload['event'] == 'batch_processed':
            self.monitor_container.batch_processed(payload['batch_size'])
        elif payload['event'] == 'task_processed':
            self.monitor_container.task_processed()

