import uuid

__all__ = ["SparseSource", "SparseStream", "SparseSink", "SparseOperator"]

class SparseStream:
    def __init__(self, protocol, no_samples, target_latency, use_scheduling):
        self.id = str(uuid.uuid4())

        self.target_latency = target_latency
        self.use_scheduling = use_scheduling
        self.no_samples = no_samples

        self.protocol = protocol

    def emit(self, data_tuple):
        self.no_samples -= 1
        payload = {'stream_id': self.id, 'activation': data_tuple}
        self.protocol.send_payload(payload)

class SparseSource:
    def __init__(self, stream : SparseStream):
        self.id = str(uuid.uuid4())
        self.stream = stream

    def get_tuple(self):
        pass

    def emit(self):
        self.stream.emit(self.get_tuple())

class SparseSink:
    def __init__(self, logger):
        self.logger = logger

    def tuple_received(self, new_tuple):
        pass

class SparseOperator:
    def __init__(self):
        self.id = str(uuid.uuid4())

    def call(self, input_tuple):
        pass

