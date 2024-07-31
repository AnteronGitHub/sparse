import asyncio

from ..node import SparseSlice
from .protocols import DownstreamConnectorProtocol

class DownstreamConnectorSlice(SparseSlice):
    def __init__(self, migrator_slice, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.migrator_slice = migrator_slice

    def get_futures(self, futures):
        futures.append(self.connect_to_downstream_server())
        return futures

    async def connect_to_downstream_server(self):
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()

        while True:
            try:
                self.logger.info("Connecting to downstream server on %s:%s.", \
                                  self.config.root_server_address, \
                                  self.config.root_server_port)
                await loop.create_connection(lambda: DownstreamConnectorProtocol(on_con_lost, self.migrator_slice), \
                                             self.config.root_server_address, \
                                             self.config.root_server_port)
                await on_con_lost
                break
            except ConnectionRefusedError:
                self.logger.warn("Connection refused. Re-trying in 5 seconds.")
                await asyncio.sleep(5)

