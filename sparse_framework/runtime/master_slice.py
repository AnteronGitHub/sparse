import asyncio

from ..node import SparseSlice
from ..protocols import SparseClientProtocol

class SparseMasterSlice(SparseSlice):
    """Sparse Master Slice maintains connection to a downstream Node in Sparse cluster network.
    """
    def get_futures(self, futures):
        futures.append(self.connect_to_server(self.config.upstream_host, self.config.upstream_port))

        return futures

    async def connect_to_server(self, host, port):
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()

        while True:
            try:
                await loop.create_connection(lambda: SparseClientProtocol(on_con_lost, self), host, port)
                result = await on_con_lost
                return result
            except ConnectionRefusedError:
                self.logger.warn("Connection refused. Re-trying in 5 seconds.")
                await asyncio.sleep(5)

