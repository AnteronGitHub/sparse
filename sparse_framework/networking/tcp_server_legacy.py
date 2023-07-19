import asyncio

from . import BaseTCPServer

class TCPServerLegacy(TCPServer):
    """asyncio TCP server initialization implementation for older Python compiler versions.
    """

    def start(self):
        loop = asyncio.get_event_loop()
        coro = asyncio.start_server(self._connection_callback, self.listen_address, self.listen_port, loop=loop)
        server = loop.run_until_complete(coro)

        # Serve requests until Ctrl+C is pressed
        self.logger.info(f"TCP server listening on {self.listen_address}:{self.listen_port}")
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass

        # Close the server
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()
