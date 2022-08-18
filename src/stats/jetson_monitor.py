import importlib

from . import NetworkMonitor

class JetsonMonitor(NetworkMonitor):
    def __init__(self, if_name = 'lo'):
        super().__init__(if_name)
        if importlib.util.find_spec("jtop") is None:
            raise 'No jtop available'
        from jtop import jtop

    def get_metrics(self):
        return super().get_metrics() + ',GPU,RAM,SWAP,power_cur'

    def read_stats(self):
        time, bytes_sent, bytes_recv = super().read_stats()
        return time, bytes_sent, bytes_recv, jetson.stats['GPU'], jetson.stats['RAM'], jetson.stats['SWAP'], jetson.stats['power cur']

    def log_stats(self):
        file_logger = FileLogger()
        file_logger.log_row(self.get_metrics())
        with jtop() as jetson:
            while jetson.ok():
                try:
                    file_logger.log_row(self.read_stats())
                except KeyboardInterrupt:
                    print("Stopping due to keyboard interrupt")
                    break
