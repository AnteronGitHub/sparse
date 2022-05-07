import asyncio

from .rx_pipe import RXPipe

class RXPipeLatest(RXPipe):
    """RX pipe implementation using the current asyncio interface.
    """

    async def receive_task(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        """Generic callback function which passes offloaded task directly to the task executor.

        """
        input_data = await reader.read()

        result_data = self.task_executor.execute_task(input_data)

        writer.write(result_data)
        await writer.drain()
        writer.close()
        print("Processed task")

    async def serve(self):
        server = await asyncio.start_server(self.receive_task, self.listen_address, self.listen_port)
        print(f'RX pipe listening on {self.listen_address}:{self.listen_port}')
        async with server:
            await server.serve_forever()

    def start(self):
        asyncio.run(self.serve())
