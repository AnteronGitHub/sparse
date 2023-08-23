from .base_monitor import BaseMonitor

class MonitorContainer(BaseMonitor):
    def __init__(self, monitors : list):
        self.monitors = monitors

    def get_metrics(self):
        metrics = []
        for monitor in self.monitors:
            metrics += monitor.get_metrics()
        return metrics

    def get_stats(self):
        stats = []
        for monitor in self.monitors:
            stats += monitor.get_stats()
        return stats

    def receive_message(self, payload : dict):
        pass
