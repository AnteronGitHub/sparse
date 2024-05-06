import asyncio
import multiprocessing

from ..node import SparseSlice
from ..protocols import SparseServerProtocol

from .io_buffer import SparsePytorchIOBuffer
from .task_executor import SparseTaskExecutor

class SparseStreamRuntimeSlice(SparseSlice):
    """Sparse Stream Runtime Slice maintains task executor, and the associated memory manager, for executing stream
    application operations locally.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = None
        self.io_buffer = None

    def get_futures(self, futures):
        m = multiprocessing.Manager()
        lock = m.Lock()

        task_queue = asyncio.Queue()

        self.io_buffer = SparsePytorchIOBuffer()
        self.executor = SparseTaskExecutor(lock, self.io_buffer, task_queue)

        futures.append(self.executor.start())
        futures.append(self.start_server(self.config.listen_address, self.config.listen_port))

        return futures

    def add_operator(self, operator):
        self.executor.add_operator(operator)

    def sync_received(self, protocol, stream_id, sync):
        self.logger.debug(f"Received {sync} s sync")
        if self.source is not None:
            if (self.source.no_samples > 0):
                offload_latency = protocol.request_statistics.get_offload_latency(protocol.current_record)

                if not self.source.use_scheduling:
                    sync = 0.0

                target_latency = self.source.target_latency

                loop = asyncio.get_running_loop()
                loop.call_later(target_latency-offload_latency + sync if target_latency > offload_latency else 0, self.source.emit)
            else:
                protocol.transport.close()

    async def start_server(self, addr, port):
        loop = asyncio.get_running_loop()

        self.logger.info(f"Data plane listening to '{addr}:{port}'")
        server = await loop.create_server(lambda: SparseServerProtocol(self), addr, port)
        async with server:
            await server.serve_forever()

