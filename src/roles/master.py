import asyncio

class TaskDeployer:
    """Class that handles network connections to available worker nodes.
    """

    def __init__(self, upstream_host : str = '127.0.0.1', upstream_port : int = 50007):
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        print(f"Using upstream {self.upstream_host}:{self.upstream_port}")

    async def stream_task_synchronous(self, input_data : bytes):
        reader, writer = await asyncio.open_connection(self.upstream_host, self.upstream_port)

        writer.write(input_data)
        writer.write_eof()
        await writer.drain()

        result_data = await reader.read()
        writer.close()

        return result_data

    def deploy_task(self, input_data : bytes):
        return asyncio.run(self.stream_task_synchronous(input_data))

class Master:
    def __init__(self, task_deployer : TaskDeployer = None):
        self.task_deployer = task_deployer or TaskDeployer()
