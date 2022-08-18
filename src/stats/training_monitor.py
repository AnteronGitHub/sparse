from .network_monitor import NetworkMonitor

class TrainingMonitor(NetworkMonitor):
    def __init__(self, if_name = 'lo'):
        super().__init__()
        self.processed_samples = 0

    def get_metrics(self):
        return super().get_metrics() + ',samples'

    def read_stats(self, newly_processed_samples = 0):
        time, bytes_sent, bytes_recv = super().read_stats()

        self.processed_samples += newly_processed_samples

        return time, bytes_sent, bytes_recv, self.processed_samples

