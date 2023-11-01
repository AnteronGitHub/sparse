from sparse_framework.networking.protocols import SparseProtocol
from sparse_framework.stats import ServerRequestStatistics

class ModelServeServerProtocol(SparseProtocol):
    def __init__(self, node, queue, stats_queue):
        super().__init__()

        self.queue = queue
        self.stats_queue = stats_queue
        self.model_repository = node.get_model_repository()
        self.statistics = ServerRequestStatistics(node.node_id, stats_queue)

        self.model_meta_data = None

    def initialize_stream(self, input_data):
        self.statistics.create_record("initialize_stream")
        self.model_meta_data = input_data['model_meta_data']
        load_task = self.model_repository.get_load_task(self.model_meta_data)
        if load_task is None:
            self.model_repository.load_model(self.model_meta_data, self.model_loaded)
        elif not load_task.done():
            load_task.add_done_callback(self.model_loaded)
        else:
            self.model_loaded(load_task)

        self.statistics.current_record.queued()

    def model_loaded(self, load_task):
        self.send_payload({ "statusCode": 200 })

    def offload_task(self, input_data):
        self.statistics.create_record("offload_task")

        load_task = self.model_repository.get_load_task(self.model_meta_data)
        model, loss_fn, optimizer = load_task.result()
        task_data = {
                'activation': self.model_repository.transferToDevice(input_data['activation']),
                'model': model
        }

        self.statistics.current_record.queued()
        self.queue.put_nowait(("forward_propagate", task_data, self.forward_propagated))

    def forward_propagated(self, result):
        task_latency = result["latency"]
        self.statistics.current_record.set_task_latency(task_latency)

        self.send_payload({ "pred": self.model_repository.transferToHost(result["pred"]) })

    def payload_received(self, payload):
        if payload["op"] == "initialize_stream":
            self.initialize_stream(payload)
        else:
            self.offload_task(payload)

    def connection_made(self, transport):
        self.statistics.connected()

        super().connection_made(transport)

    def connection_lost(self, exc):
        self.logger.info(self.statistics)

        super().connection_lost(exc)

    def send_payload(self, payload):
        super().send_payload(payload)

        self.statistics.task_completed()
