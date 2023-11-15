import asyncio

from sparse_framework import SparseNode

from .executor import TensorExecutor
from .protocols import InferenceClientProtocol, InferenceServerProtocol, ParameterServerProtocol
from .memory_buffer import MemoryBuffer
from .utils import get_device

__all__ = ["InferenceClient", "InferenceServer", "ParameterServer"]

class InferenceClient(SparseNode):
    """A Node that iterates over a dataset and offloads the sample inference to specified server.
    """
    def __init__(self, dataset, model_meta_data, no_samples, use_scheduling, **kwargs):
        super().__init__(**kwargs)
        self.protocol_factory = lambda on_con_lost, stats_queue: \
                                        lambda: InferenceClientProtocol(self.node_id, \
                                                                        dataset, \
                                                                        model_meta_data, \
                                                                        on_con_lost, \
                                                                        no_samples, \
                                                                        use_scheduling, \
                                                                        stats_queue=stats_queue)


    def get_futures(self):
        futures = super().get_futures()

        futures.append(self.connect_to_server(self.protocol_factory,
                                              self.config.upstream_host,
                                              self.config.upstream_port,
                                              self.result_callback))

        return futures

    def result_callback(self, result):
        self.logger.info(result)

class InferenceServer(SparseNode):
    """A Node serves inference requests for models over a TCP connection.

    Inference server runs the executor in a separate thread from the server, using an asynchronous queue.
    """
    def __init__(self):
        super().__init__()

        self.memory_buffer = None

    def get_memory_buffer(self) -> MemoryBuffer:
        if (self.memory_buffer is None):
            self.memory_buffer = MemoryBuffer(self, get_device())
        return self.memory_buffer

    def get_futures(self):
        futures = super().get_futures()

        task_queue = asyncio.Queue()

        futures.append(TensorExecutor(task_queue).start())
        futures.append(self.start_server(lambda: InferenceServerProtocol(self, task_queue, self.stats_queue), \
                                         self.config.listen_address, \
                                         self.config.listen_port))

        return futures

class ParameterServer(SparseNode):
    """A Node serves model parameters over a TCP connection.
    """
    def get_futures(self):
        futures = super().get_futures()

        futures.append(self.start_server(lambda: ParameterServerProtocol(), \
                                         self.config.model_server_address, \
                                         self.config.model_server_port))

        return futures
