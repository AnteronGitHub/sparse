from time import time

__all__ = ["RequestStatistics", "ClientRequestStatistics", "ServerRequestStatistics", "RequestStatisticsRecord"]

class RequestStatisticsRecord:
    def __init__(self, node_id : str, request_op : str, connection_made_at : float):
        self.node_id = node_id
        self.request_op = request_op
        self.connection_made_at = connection_made_at
        self.processing_started = time() - self.connection_made_at
        self.completed_at = None

    def completed(self):
        self.completed_at = time() - self.connection_made_at

    def get_latency(self):
        return self.completed_at - self.processing_started

    def csv_header(self):
        return "node_id,request_op,processing_started,latency\n"

    def to_csv(self):
        return f"{self.node_id},{self.request_op},{self.processing_started},{self.get_latency()}\n"

class ServerRequestStatisticsRecord(RequestStatisticsRecord):
    def __init__(self, *args):
        super().__init__(*args)
        self.task_queued = None
        self.task_latency = 0.0
        self.serialization_latency = 0.0
        self.deserialization_latency = 0.0

    def queued(self, deserialization_latency = 0.0):
        self.task_queued = time() - self.connection_made_at
        self.deserialization_latency = deserialization_latency

    def set_task_latency(self, task_latency):
        self.task_latency = task_latency

    def set_serialization_latency(self, serialization_latency):
        self.serialization_latency = serialization_latency

    def get_queuing_time(self):
        return self.completed_at - self.task_queued - self.task_latency

    def csv_header(self):
        return "node_id,request_op,processing_started,latency,queuing_time,task_latency\n"

    def to_csv(self):
        return f"{self.node_id},{self.request_op},{self.processing_started},{self.get_latency()},{self.get_queuing_time()},{self.task_latency}\n"

class ClientRequestStatisticsRecord(RequestStatisticsRecord):
    def __init__(self, *args):
        super().__init__(*args)
        self.request_sent = None

    def sent(self):
        self.request_sent = time() - self.connection_made_at

    def get_offload_latency(self):
        return self.completed_at - self.request_sent

    def csv_header(self):
        return "node_id,request_op,processing_started,latency,offload_latency\n"

    def to_csv(self):
        return f"{self.node_id},{self.request_op},{self.processing_started},{self.get_latency()},{self.get_offload_latency()}\n"

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
            return 0, 0, 0, 0, 0, 0

        avg_latency = 0.0
        avg_queuing_time = 0.0
        avg_task_latency = 0.0
        avg_serialization_latency = 0.0
        avg_deserialization_latency = 0.0
        for record in records:
            avg_latency += record.get_latency()
            avg_queuing_time += record.get_queuing_time()
            avg_task_latency += record.task_latency
            avg_serialization_latency += record.serialization_latency
            avg_deserialization_latency += record.deserialization_latency

        avg_latency /= no_requests
        avg_queuing_time /= no_requests
        avg_task_latency /= no_requests
        avg_serialization_latency /= no_requests
        avg_deserialization_latency /= no_requests

        return no_requests, avg_latency, avg_queuing_time, avg_task_latency, avg_serialization_latency, avg_deserialization_latency

    def __str__(self):
        if len(self.request_records) == 0:
            return "No requests made during connection."
        else:
            no_requests, avg_latency, avg_queuing_time, avg_task_latency, avg_serialization_latency, avg_deserialization_latency = self.count_statistics()
            return f"{no_requests} tasks / {1000.0*avg_latency:.2f} ms avg e2e lat / {1000*avg_task_latency:.2f} ms avg task latency / {1000*avg_queuing_time:.2f} ms avg queuing / {1000.0*avg_serialization_latency:.2f} ms serialization / {1000.0*avg_deserialization_latency:.2f} ms deserialization"

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

