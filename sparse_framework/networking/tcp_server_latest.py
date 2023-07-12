import asyncio

from .base_tcp_server import BaseTCPServer

class TCPServerLatest(BaseTCPServer):
    """TCP server implementation using the current asyncio interface.
    """

    async def serve(self):
        server = await asyncio.start_server(self._connection_callback, self.listen_address, self.listen_port)
        self.logger.info(f"TCP server listening on {self.listen_address}:{self.listen_port}")
        await server.serve_forever()

    def start(self):
        asyncio.run(self.serve())
