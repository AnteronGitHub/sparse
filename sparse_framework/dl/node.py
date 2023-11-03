import asyncio

from sparse_framework import Master, Worker

from .executor import TensorExecutor
from .protocols import ModelServeClientProtocol, ModelServeServerProtocol
from .memory_buffer import MemoryBuffer
from .utils import get_device

__all__ = ["ModelServeClient", "ModelServeServer"]

class ModelServeClient(Master):
    def __init__(self, dataset, model_meta_data, no_samples, **kwargs):
        super().__init__(**kwargs)
        self.protocol_factory = lambda on_con_lost, stats_queue: lambda: ModelServeClientProtocol(self.node_id, dataset, model_meta_data, on_con_lost, no_samples, stats_queue=stats_queue)

    def get_futures(self):
        futures = super().get_futures()

        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()
        futures.append(loop.create_connection(self.protocol_factory(on_con_lost, self.stats_queue),
                                              self.config_manager.upstream_host,
                                              self.config_manager.upstream_port))
        futures.append(on_con_lost)

        return futures

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

