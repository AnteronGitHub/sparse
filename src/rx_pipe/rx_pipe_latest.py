import asyncio

from . import RXPipe

class RXPipeLatest(RXPipe):
    """RX pipe implementation using the current asyncio interface.
    """

    async def serve(self):
        server = await asyncio.start_server(self.receive_task, self.listen_address, self.listen_port)
        self.logger.info(f"RX pipe listening on {self.listen_address}:{self.listen_port}")
        await server.serve_forever()

    def start(self):
        self.logger.debug("Starting RX pipe")
        asyncio.run(self.serve())
