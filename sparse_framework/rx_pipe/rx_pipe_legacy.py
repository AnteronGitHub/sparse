import asyncio

from . import RXPipe

class RXPipeLegacy(RXPipe):
    """Legacy asyncio implementation for older Python compiler versions.
    """

    def start(self):
        self.logger.debug("Starting RX pipe")
        loop = asyncio.get_event_loop()
        coro = asyncio.start_server(self.receive_task, self.listen_address, self.listen_port, loop=loop)
        server = loop.run_until_complete(coro)

        # Serve requests until Ctrl+C is pressed
        self.logger.info(f"RX pipe listening on {self.listen_address}:{self.listen_port}")
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass

        # Close the server
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()
