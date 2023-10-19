from time import time

from .request_statistics_record import RequestStatisticsRecord

class RequestStatistics():
    def __init__(self, node_id : str, stats_queue = None):
        self.node_id = node_id
        self.request_records = []
        self.stats_queue = stats_queue

        self.connected_at = None
        self.request_record = None

    def connected(self):
        self.connected_at = time()

    def task_started(self, task_op):
        self.request_record = RequestStatisticsRecord(self.node_id, task_op, self.connected_at)

    def request_sent(self):
        self.request_record.sent()

    def task_completed(self):
        self.request_record.completed()
        self.log_record()
        return self.request_record.get_latency()

    def log_record(self):
        self.request_records.append(self.request_record)

        if self.stats_queue is not None:
            self.stats_queue.put_nowait((self.request_record))

    def count_statistics(self):
        no_requests = len(self.request_records)
        if no_requests == 0:
            return 0, 0, 0, 0

        avg_latency = 0
        avg_offload_latency = 0
        avg_ratio = 0
        for record in self.request_records:
            avg_latency += record.get_latency()
            avg_offload_latency += record.get_offload_latency()
            avg_ratio += record.get_offload_latency() / record.get_latency()

        avg_latency /= no_requests
        avg_offload_latency /= no_requests
        avg_ratio /= no_requests

        return no_requests, avg_latency, avg_offload_latency, avg_ratio

    def print_statistics(self):
        no_requests, avg_latency, avg_offload_latency, avg_ratio = self.count_statistics()
        if avg_latency == 0:
            return "No requests made during connection."
        else:
            return f"{no_requests} tasks / {1.0/avg_latency:.2f} avg FPS / {1000*avg_offload_latency:.2f} ms avg request latency / {100.0*avg_ratio:.2f} % avg offload latency ratio."
