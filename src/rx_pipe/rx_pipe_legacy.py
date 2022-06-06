import asyncio

from . import RXPipe

class RXPipeLegacy(RXPipe):
    """Legacy asyncio implementation for older Python compiler versions.
    """

    async def receive_task(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        """Generic callback function which passes offloaded task directly to the task executor.

        """
        input_data = await reader.read()

        result_data = await self.task_executor.execute_task(input_data)

        writer.write(result_data)
        await writer.drain()
        writer.close()
        self.logger.debug("Processed task")

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
