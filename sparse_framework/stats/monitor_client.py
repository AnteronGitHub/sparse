import uuid

from sparse_framework.networking import UnixSocketClient

class MonitorClient(UnixSocketClient):
    def __init__(self, socket_path = '/run/sparse/sparse-benchmark.sock'):
        UnixSocketClient.__init__(self, socket_path)

        self.monitor_benchmark_id = None
        self.client_benchmark_id = None

    def start_benchmark(self, log_file_prefix = 'benchmark_sparse', benchmark_type = 'MonitorBenchmark'):
        if benchmark_type == "MonitorBenchmark" and self.monitor_benchmark_id:
            return
        if benchmark_type == "ClientBenchmark" and self.client_benchmark_id:
            return

        benchmark_id = str(uuid.uuid4())
        if benchmark_type == "MonitorBenchmark":
            self.monitor_benchmark_id = benchmark_id
        elif benchmark_type == "ClientBenchmark":
            self.client_benchmark_id = benchmark_id

        self.submit_event({ "benchmark_id": benchmark_id,
                            "event": "start",
                            "benchmark_type": benchmark_type,
                            "log_file_prefix": log_file_prefix })

    def stop_benchmark(self):
        if self.monitor_benchmark_id:
            self.submit_event({ "benchmark_id": self.monitor_benchmark_id,
                                "event": "stop_benchmark" })
        if self.client_benchmark_id:
            self.submit_event({ "benchmark_id": self.client_benchmark_id,
                                "event": "stop_benchmark" })

    def batch_processed(self, batch_size : int, loss : float = None):
        if self.monitor_benchmark_id:
            self.submit_event({ "benchmark_id": self.monitor_benchmark_id,
                                "event": "batch_processed",
                                "batch_size": batch_size,
                                "loss": loss })

    def task_completed(self, processing_time : float = None):
        if self.monitor_benchmark_id:
            self.submit_event({ "benchmark_id": self.monitor_benchmark_id,
                                "event": "task_completed" })
        if self.client_benchmark_id:
            self.submit_event({ "benchmark_id": self.client_benchmark_id,
                                "event": "task_completed",
                                "processing_time": processing_time })

    def connection_timeout(self):
        if self.monitor_benchmark_id:
            self.submit_event({ "benchmark_id": self.monitor_benchmark_id,
                                "event": "connection_timeout" })

    def broken_pipe_error(self):
        if self.monitor_benchmark_id:
            self.submit_event({ "benchmark_id": self.monitor_benchmark_id,
                                "event": "broken_pipe_error" })

