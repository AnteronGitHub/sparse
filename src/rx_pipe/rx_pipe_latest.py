import asyncio

from . import RXPipe

class RXPipeLatest(RXPipe):
    """RX pipe implementation using the current asyncio interface.
    """

    async def receive_task(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        """Generic callback function which passes offloaded task directly to the task executor.

        """
        input_data = await reader.read()

        result_data = await self.task_executor.execute_task(input_data)

        writer.write(result_data)
        await writer.drain()
        writer.close()
        self.logger.info("Processed task")

    async def serve(self):
        server = await asyncio.start_server(self.receive_task, self.listen_address, self.listen_port)
        self.logger.info(f"RX pipe listening on {self.listen_address}:{self.listen_port}")
        async with server:
            await server.serve_forever()

    def start(self):
        self.logger.debug("Starting RX pipe")
        asyncio.run(self.serve())
