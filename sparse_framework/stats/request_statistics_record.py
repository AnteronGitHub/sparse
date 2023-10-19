from time import time

class RequestStatisticsRecord:
    def __init__(self, node_id : str, request_op : str, connection_made_at : float):
        self.node_id = node_id
        self.request_op = request_op
        self.connection_made_at = connection_made_at
        self.processing_started = time() - self.connection_made_at
        self.request_sent = None
        self.completed_at = None

    def sent(self):
        self.request_sent = time() - self.connection_made_at

    def completed(self):
        self.completed_at = time() - self.connection_made_at

    def get_latency(self):
        return self.completed_at - self.processing_started

    def get_offload_latency(self):
        return self.completed_at - self.request_sent

    @staticmethod
    def csv_header():
        return "node_id,request_op,processing_started,latency,offload_latency\n"

    def to_csv(self):
        return f"{self.node_id},{self.request_op},{self.processing_started},{self.get_latency()},{self.get_offload_latency()}\n"

