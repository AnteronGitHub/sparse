import psutil

from . import Monitor

class NetworkMonitor(Monitor):
    def __init__(self, nic = ""):
        self.initial_bytes_sent = None
        self.initial_bytes_recv = None
        self.nic = nic

    def get_metrics(self):
        return ['bytes_sent', 'bytes_recv']

    def get_stats(self):
        pernic = len(self.nic) > 0
        network_stats = psutil.net_io_counters(pernic=pernic)
        if pernic:
            network_stats = network_stats[self.nic]

        if self.initial_bytes_sent is None:
            self.initial_bytes_sent = network_stats[0]
        if self.initial_bytes_recv is None:
            self.initial_bytes_recv = network_stats[1]

        return [network_stats[0] - self.initial_bytes_sent, network_stats[1] - self.initial_bytes_recv]

