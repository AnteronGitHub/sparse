import importlib
from jtop import jtop

from .network_monitor import NetworkMonitor

class JetsonMonitor(NetworkMonitor):
    def __init__(self, if_name = 'lo'):
        super().__init__(if_name)
        self.jetson = None

    def get_metrics(self):
        return super().get_metrics() + ',GPU,RAM,SWAP,power_cur'

    def read_stats(self):
        time, bytes_sent, bytes_recv = super().read_stats()

        if self.jetson is None:
            with jtop() as jetson:
                if jetson.ok():
                    stats = jetson.stats
        else:
            stats = self.jetson.stats

        return time, bytes_sent, bytes_recv, stats['GPU'], stats['RAM'], stats['SWAP'], stats['power cur']

    def log_stats(self):
        file_logger = FileLogger()
        file_logger.log_row(self.get_metrics())
        with jtop() as jetson:
            self.jetson = jetson
            while jetson.ok():
                try:
                    file_logger.log_row(self.read_stats())
                except KeyboardInterrupt:
                    print("Stopping due to keyboard interrupt")
                    break

        self.jetson = None
