from time import time

from .request_statistics_record import ClientRequestStatisticsRecord, RequestStatisticsRecord, ServerRequestStatisticsRecord

class RequestStatistics():
    def __init__(self, node_id : str, stats_queue = None):
        self.node_id = node_id
        self.request_records = []
        self.stats_queue = stats_queue

        self.connected_at = None
        self.current_record = None

    def connected(self):
        self.connected_at = time()

    def create_record(self, task_op):
        self.current_record = RequestStatisticsRecord(self.node_id, task_op, self.connected_at)

    def task_completed(self):
        self.current_record.completed()
        self.request_records.append(self.current_record)

        if self.stats_queue is not None:
            self.stats_queue.put_nowait((self.current_record))

        return self.current_record.get_latency()

    def count_statistics(self):
        records = [r for r in self.request_records if r.request_op != "initialize_stream"]
        no_requests = len(records)
        if no_requests == 0:
            return 0, 0

        avg_latency = 0
        for record in records:
            avg_latency += record.get_latency()

        avg_latency /= no_requests

        return no_requests, avg_latency

    def __str__(self):
        if len(self.request_records) == 0:
            return "No requests made during connection."
        else:
            no_requests, avg_latency = self.count_statistics()
            return f"{no_requests} tasks / {1.0/avg_latency:.2f} avg FPS."

class ServerRequestStatistics(RequestStatistics):
    def create_record(self, task_op):
        self.current_record = ServerRequestStatisticsRecord(self.node_id, task_op, self.connected_at)

    def count_statistics(self):
        records = [r for r in self.request_records if r.request_op != "initialize_stream"]
        no_requests = len(records)
        if no_requests == 0:
            return 0, 0, 0, 0

        avg_latency = 0
        avg_queuing_time = 0
        avg_task_latency = 0
        for record in records:
            avg_latency += record.get_latency()
            avg_queuing_time += record.get_queuing_time()
            avg_task_latency += record.task_latency

        avg_latency /= no_requests
        avg_queuing_time /= no_requests
        avg_task_latency /= no_requests

        return no_requests, avg_latency, avg_queuing_time, avg_task_latency

    def __str__(self):
        if len(self.request_records) == 0:
            return "No requests made during connection."
        else:
            no_requests, avg_latency, avg_queuing_time, avg_task_latency = self.count_statistics()
            return f"{no_requests} tasks / {1.0/avg_latency:.2f} avg FPS / {1000*avg_queuing_time:.2f} ms avg queuing time / {1000*avg_task_latency:.2f} ms avg task latency"

class ClientRequestStatistics(RequestStatistics):
    def create_record(self, task_op):
        self.current_record = ClientRequestStatisticsRecord(self.node_id, task_op, self.connected_at)

    def request_sent(self):
        self.current_record.sent()

    def count_statistics(self):
        records = [r for r in self.request_records if r.request_op != "initialize_stream"]
        no_requests = len(records)
        if no_requests == 0:
            return 0, 0, 0, 0

        avg_latency = 0
        avg_offload_latency = 0
        avg_ratio = 0
        for record in records:
            avg_latency += record.get_latency()
            avg_offload_latency += record.get_offload_latency()
            avg_ratio += record.get_offload_latency() / record.get_latency()

        avg_latency /= no_requests
        avg_offload_latency /= no_requests
        avg_ratio /= no_requests

        return no_requests, avg_latency, avg_offload_latency, avg_ratio

    def __str__(self):
        if len(self.request_records) == 0:
            return "No requests made during connection."
        else:
            no_requests, avg_latency, avg_offload_latency, avg_ratio = self.count_statistics()
            return f"{no_requests} tasks / {1.0/avg_latency:.2f} avg FPS / {1000*avg_offload_latency:.2f} ms avg request latency / {100.0*avg_ratio:.2f} % avg offload latency ratio."

