from time import time

__all__ = ["RequestStatistics", "ClientRequestStatistics", "ServerRequestStatistics", "RequestStatisticsRecord"]

class RequestStatisticsRecord:
    """Class used for collecting timestamps from request processing.

    The timestamps are collected relative to the time at which the connection was made.
    """
    def __init__(self, connection_id : str, request_op : str, connection_made_at : float):
        self.connection_id = connection_id
        self.request_op = request_op
        self.connection_made_at = connection_made_at

    def csv_header(self):
        return "connection_id,request_op\n"

    def to_csv(self):
        return f"{self.connection_id},{self.request_op}\n"

class ServerRequestStatisticsRecord(RequestStatisticsRecord):
    def __init__(self, *args):
        super().__init__(*args)
        self.request_received_at = None
        self.task_queued_at = None
        self.task_started_at = None
        self.batch_no = None
        self.task_completed_at = None
        self.response_sent_at = None
        self.sync_delay_ms = 0.0

    def request_received(self):
        self.request_received_at = time() - self.connection_made_at

    def task_queued(self):
        self.task_queued_at = time() - self.connection_made_at

    def task_started(self, task_started_at = None, batch_no = 0):
        if task_started_at is None:
            task_started_at = time()
        self.task_started_at = task_started_at - self.connection_made_at
        self.batch_no = batch_no

    def task_completed(self, task_completed_at = None):
        if task_completed_at is None:
            task_completed_at = time()
        self.task_completed_at = task_completed_at - self.connection_made_at

    def set_sync_delay_ms(self, sync_delay_ms):
        self.sync_delay_ms = sync_delay_ms

    def response_sent(self):
        self.response_sent_at = time() - self.connection_made_at

    def csv_header(self):
        return "connection_id,request_op,request_received_at,task_queued_at,task_started_at,task_completed_at,response_sent_at,batch_no,sync_delay_ms\n"

    def to_csv(self):
        return f"{self.connection_id},{self.request_op},{self.request_received_at},{self.task_queued_at},{self.task_started_at},{self.task_completed_at},{self.response_sent_at},{self.batch_no},{self.sync_delay_ms}\n"

class ClientRequestStatisticsRecord(RequestStatisticsRecord):
    def __init__(self, *args):
        super().__init__(*args)
        self.processing_started_at = None
        self.request_sent_at = None
        self.response_received_at = None

    def processing_started(self):
        self.processing_started_at = time() - self.connection_made_at

    def request_sent(self):
        self.request_sent_at = time() - self.connection_made_at

    def response_received(self):
        self.response_received_at = time() - self.connection_made_at

    def csv_header(self):
        return "connection_id,request_op,processing_started_at,request_sent_at,response_received_at\n"

    def to_csv(self):
        return f"{self.connection_id},{self.request_op},{self.processing_started_at},{self.request_sent_at},{self.response_received_at}\n"

class RequestStatistics():
    """Class that collects statistics for requests made during a connection.
    """
    def __init__(self, connection_id : str, stats_queue = None):
        self.connection_id = connection_id
        self.request_records = []
        self.stats_queue = stats_queue

        self.connected_at = None

    def connected(self):
        self.connected_at = time()

    def create_record(self, task_op) -> RequestStatisticsRecord:
        return RequestStatisticsRecord(self.connection_id, task_op, self.connected_at)

    def log_record(self, record : RequestStatisticsRecord):
        self.request_records.append(record)

        if self.stats_queue is not None:
            self.stats_queue.put_nowait((record))

    def __str__(self):
        no_requests = len(self.request_records)
        if no_requests == 0:
            return "No requests made during connection."
        else:
            return f"{no_requests} tasks."

class ServerRequestStatistics(RequestStatistics):
    def create_record(self, task_op) -> ServerRequestStatisticsRecord:
        return ServerRequestStatisticsRecord(self.connection_id, task_op, self.connected_at)

    def get_service_time(self, record : ServerRequestStatisticsRecord):
        return record.response_sent_at - record.request_received_at

    def get_queueing_time(self, record : ServerRequestStatisticsRecord):
        return record.task_started_at - record.task_queued_at

    def get_task_latency(self, record : ServerRequestStatisticsRecord):
        return record.task_completed_at - record.task_started_at

    def count_offload_task_statistics(self):
        records = [r for r in self.request_records if r.request_op == "offload_task"]
        no_requests = len(records)
        if no_requests == 0:
            return 0, None, None, None

        avg_service_time = 0.0
        avg_queuing_time = 0.0
        avg_task_latency = 0.0
        for record in records:
            avg_service_time += self.get_service_time(record)
            avg_queuing_time += self.get_queueing_time(record)
            avg_task_latency += self.get_task_latency(record)

        avg_service_time /= no_requests
        avg_queuing_time /= no_requests
        avg_task_latency /= no_requests

        return no_requests, avg_service_time, avg_queuing_time, avg_task_latency

    def __str__(self):
        no_requests, avg_service_time, avg_queuing_time, avg_task_latency = self.count_offload_task_statistics()
        if no_requests == 0:
            return "No requests made during connection."
        else:
            return f"{no_requests} tasks / {1000.0*avg_service_time:.2f} ms avg service time / {1000*avg_task_latency:.2f} ms avg task latency / {1000*avg_queuing_time:.2f} ms avg queuing"

class ClientRequestStatistics(RequestStatistics):
    def create_record(self, task_op) -> ClientRequestStatisticsRecord:
        return ClientRequestStatisticsRecord(self.connection_id, task_op, self.connected_at)

    def get_e2e_latency(self, record : ClientRequestStatisticsRecord):
        return record.response_received_at - record.processing_started_at

    def get_offload_latency(self, record : ClientRequestStatisticsRecord):
        return record.response_received_at - record.request_sent_at

    def count_offload_task_statistics(self):
        records = [r for r in self.request_records if r.request_op == "offload_task"]
        no_requests = len(records)
        if no_requests == 0:
            return 0, None, None

        avg_e2e_latency = 0.0
        avg_offload_latency = 0.0

        for record in records:
            avg_e2e_latency += self.get_e2e_latency(record)
            avg_offload_latency += self.get_offload_latency(record)

        avg_e2e_latency /= no_requests
        avg_offload_latency /= no_requests

        return no_requests, avg_e2e_latency, avg_offload_latency

    def __str__(self):
        no_requests, avg_e2e_latency, avg_offload_latency = self.count_offload_task_statistics()
        if no_requests == 0:
            return "No requests made during connection."
        else:
            return f"{no_requests} tasks / {1.0/avg_e2e_latency:.2f} FPS avg sample rate / {1000.0*avg_offload_latency:.2f} ms avg offload latency."

