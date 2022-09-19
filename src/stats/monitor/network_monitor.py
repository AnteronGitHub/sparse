import psutil

from . import Monitor

class NetworkMonitor(Monitor):
    def __init__(self):
        self.initial_bytes_sent = None
        self.initial_bytes_recv = None

    def get_metrics(self):
        return ['bytes_sent', 'bytes_recv']

    def get_stats(self):
        network_stats = psutil.net_io_counters()

        if self.initial_bytes_sent is None:
            self.initial_bytes_sent = network_stats[0]
        if self.initial_bytes_recv is None:
            self.initial_bytes_recv = network_stats[1]

        return [network_stats[0] - self.initial_bytes_sent, network_stats[1] - self.initial_bytes_recv]

