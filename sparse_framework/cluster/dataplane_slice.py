from ..node import SparseSlice
from .protocols import DataPlaneProtocol

class DataPlaneSlice(SparseSlice):
    """Sparse Data Plane Slice maintains connections to downstream nodes.
    """
    def __init__(self, runtime, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime = runtime

    def get_futures(self, futures):
        futures.append(self.start_server(self.config.listen_address, self.config.listen_port))

        return futures

    async def start_server(self, addr, port):
        loop = asyncio.get_running_loop()

        self.logger.info(f"Data plane listening to '{addr}:{port}'")
        server = await loop.create_server(lambda: DataPlaneProtocol(self), addr, port)
        async with server:
            await server.serve_forever()

