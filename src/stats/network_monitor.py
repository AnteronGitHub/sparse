import psutil

from . import Monitor

class NetworkMonitor(Monitor):
    def __init__(self, if_name = 'lo'):
        super().__init__()
        self.if_name = if_name
        self.initial_bytes_sent = self.initial_bytes_recv = None

    def get_metrics(self):
        return super().get_metrics() + ',bytes_sent,bytes_recv'

    def read_stats(self):
        time = super().get_stats()
        nic_stats = psutil.net_io_counters(pernic=True)
        [bytes_sent, bytes_recv, packets_sent, packets_recv, errin, errout, dropin, dropout] = nic_stats[self.if_name]
        if self.initial_bytes_sent is None:
            self.initial_bytes_sent = bytes_sent
        if self.initial_bytes_recv is None:
            self.initial_bytes_recv = bytes_recv

        return time, bytes_sent - self.initial_bytes_sent, bytes_recv - self.initial_bytes_recv

