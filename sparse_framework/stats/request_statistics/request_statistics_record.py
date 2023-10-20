from time import time

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

    def queued(self):
        self.task_queued = time() - self.connection_made_at

    def set_task_latency(self, task_latency):
        self.task_latency = task_latency

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

