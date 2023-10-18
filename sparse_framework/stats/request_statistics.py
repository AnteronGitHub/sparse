from .request_statistics_record import RequestStatisticsRecord

class RequestStatistics():
    def __init__(self):
        self.request_records = []

    def add_record(self, latency, offload_latency):
        self.request_records.append(RequestStatisticsRecord(latency, offload_latency))

    def count_statistics(self):
        no_requests = len(self.request_records)
        if no_requests == 0:
            return 0, 0, 0, 0

        avg_latency = 0
        avg_offload_latency = 0
        avg_ratio = 0
        for record in self.request_records:
            avg_latency += record.latency
            avg_offload_latency += record.offload_latency
            avg_ratio += record.offload_latency / record.latency

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
