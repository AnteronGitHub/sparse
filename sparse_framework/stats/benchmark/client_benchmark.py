from .base_benchmark import BaseBenchmark

class ClientBenchmark(BaseBenchmark):
    """Benchmark class that gets all the statistics from client messages.
    """

    def start(self):
        super().start()

        self.write_log(["processing_time"])

    def receive_message(self, payload : dict):
        super().receive_message(payload)

        if payload['event'] == 'stop_benchmark':
            self.stop()
        elif payload['event'] == 'task_completed':
            self.write_log([payload["processing_time"]])
