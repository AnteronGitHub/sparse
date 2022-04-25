import asyncio

from utils import get_device

class TaskExecutor:
    """Common base class for task execution logic. This class implements potentially hardware-accelerated computations
    that are offloaded into worker nodes.

    User is expected to implement the computation logic by defining a custom execute_task() function. Additionally it
    is possible to implement custom initialization code by overriding optional start() hook.
    """

    def __init__(self):
        self.device = get_device()

    def start():
        pass

    def execute_task(input_data : bytes) -> bytes:
        raise "Task executor not implemented! See documentation on how to implement your own executor"

class RXPipe:
    """Class that handles the queuing of offloaded tasks data, and passes them to the application-specific task
    executor instance.

    User is not expected to implement a custom class in normal use cases, but instead use implementations provided by
    the framework. Currently only one such implementation exists, namely this class.
    """

    def __init__(self, task_executor : TaskExecutor, listen_address : str, listen_port : int):
        self.listen_address = listen_address
        self.listen_port = listen_port
        self.task_executor = task_executor

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

class Worker:
    def __init__(self,
                 task_executor : TaskExecutor,
                 listen_address : str = '127.0.0.1',
                 listen_port : int = 50007):
        self.task_executor = task_executor
        self.rx_pipe = RXPipe(self.task_executor, listen_address, listen_port)

    def start(self):
        self.rx_pipe.start()
