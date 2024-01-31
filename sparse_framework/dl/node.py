import asyncio
import multiprocessing

from sparse_framework import SparseNode

from .executor import TensorExecutor
from .protocols import InferenceClientProtocol, InferenceServerProtocol, ParameterServerProtocol
from .memory_buffer import MemoryBuffer
from .utils import get_device

__all__ = ["InferenceClient", "InferenceServer", "ParameterServer"]

class InferenceClient(SparseNode):
    """A Node that iterates over a dataset and offloads the sample inference to specified server.
    """
    def __init__(self, dataset, model_meta_data, no_samples, use_scheduling, target_latency, **kwargs):
        super().__init__(**kwargs)
        self.protocol_factory = lambda on_con_lost, stats_queue: \
                                        lambda: InferenceClientProtocol(self.node_id, \
                                                                        dataset, \
                                                                        model_meta_data, \
                                                                        on_con_lost, \
                                                                        no_samples, \
                                                                        use_scheduling, \
                                                                        target_latency, \
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
    def __init__(self, use_scheduling : bool, use_batching : bool):
        super().__init__()

        self.use_scheduling = use_scheduling
        self.use_batching = use_batching

    def get_futures(self):
        futures = super().get_futures()

        m = multiprocessing.Manager()
        lock = m.Lock()

        memory_buffer = MemoryBuffer(self, get_device())
        task_queue = asyncio.Queue()

        futures.append(TensorExecutor(self.use_batching, lock, memory_buffer, task_queue).start())
        futures.append(self.start_server(lambda: InferenceServerProtocol(memory_buffer, \
                                                                         self.use_scheduling, \
                                                                         self.use_batching, \
                                                                         task_queue, \
                                                                         self.stats_queue,
                                                                         lock), \
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
