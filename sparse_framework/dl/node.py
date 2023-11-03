import asyncio

from sparse_framework import Master, Node, Worker

from .executor import TensorExecutor
from .protocols import ModelServeClientProtocol, ModelServeServerProtocol, ModelDownloaderServerProtocol
from .memory_buffer import MemoryBuffer
from .utils import get_device

__all__ = ["ModelServeClient", "ModelServeServer"]

class ModelServeClient(Master):
    def __init__(self, dataset, model_meta_data, no_samples, **kwargs):
        protocol_factory = lambda on_con_lost, stats_queue: \
                                lambda: ModelServeClientProtocol(self.node_id, \
                                                                 dataset, \
                                                                 model_meta_data, \
                                                                 on_con_lost, \
                                                                 no_samples, \
                                                                 stats_queue=stats_queue)

        super().__init__(protocol_factory, callback=self.result_callback, **kwargs)

    def result_callback(self, result):
        self.logger.info(result)

class ModelServeServer(Worker):
    def __init__(self):
        rx_protocol_factory = lambda task_queue, stats_queue: \
                                    lambda: ModelServeServerProtocol(self, task_queue, stats_queue)
        super().__init__(rx_protocol_factory, task_executor=TensorExecutor)

        self.memory_buffer = None

    def get_memory_buffer(self) -> MemoryBuffer:
        if (self.memory_buffer is None):
            self.memory_buffer = MemoryBuffer(self, get_device())
        return self.memory_buffer

class ModelServer(Node):
    def get_futures(self):
        futures = super().get_futures()

        futures.append(self.start_rx_pipe(lambda: ModelDownloaderServerProtocol(), \
                                          self.config_manager.model_server_address, \
                                          self.config_manager.model_server_port))

        return futures
