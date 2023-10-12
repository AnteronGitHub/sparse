import asyncio
import pickle

from .base_tcp_server import BaseTCPServer

class TCPServerLatest(BaseTCPServer):
    """TCP server implementation using the current asyncio interface.
    """

    async def serve(self, protocol_factory):
        loop = asyncio.get_running_loop()

        server = await loop.create_server(protocol_factory, self.listen_address, self.listen_port)
        async with server:
            await server.serve_forever()

