import time

from .base_benchmark import BaseBenchmark
from ..monitor import MonitorContainer

class MonitorBenchmark(BaseBenchmark):
    """Benchmark class that reads statistics by using a MonitorContainer instance.
    """
    def __init__(self, *args, monitor_container : MonitorContainer):
        super().__init__(*args)

        self.monitor_container = monitor_container

    def log_stats(self):
        self.write_log(self.monitor_container.get_stats())

    def start(self):
        super().start()

        self.write_log(self.monitor_container.get_metrics())
        self.write_log(self.monitor_container.get_stats())

    def receive_message(self, payload : dict):
        super().receive_message(payload)

        if payload['event'] == 'stop_benchmark':
            self.stop()
        else:
            self.monitor_container.receive_message(payload)
