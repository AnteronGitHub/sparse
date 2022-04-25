import asyncio

from utils import get_device

class TaskExecutor:
    def __init__(self):
        self.device = get_device()

    def start():
        pass

    def execute_task(input_data : bytes):
        raise "Method not implemented"

class Worker:
    def __init__(self, task_executor : TaskExecutor = None):
        self.task_executor = task_executor or TaskExecutor()
        self.listen_address = '127.0.0.1'
        self.listen_port = 50007

    async def process_task(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        """Callback function used for serving gradient computation requests.

        """
        input_data = await reader.read()

        result_data = self.task_executor.execute_task(input_data)

        writer.write(result_data)
        await writer.drain()
        writer.close()
        print("Processed task")

    async def serve(self):
        server = await asyncio.start_server(self.process_task, self.listen_address, self.listen_port)
        addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
        print(f'Serving on {addrs}')
        async with server:
            await server.serve_forever()

    def start(self):
        asyncio.run(self.serve())
