
class RequestStatisticsRecord():
    def __init__(self, latency : float, offload_latency : float):
        self.latency = latency
        self.offload_latency = offload_latency

