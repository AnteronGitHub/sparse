import asyncio

from .master import TaskDeployer

class TaskDeployerLatest(TaskDeployer):
    """Task deployer implementation using the current asyncio interface.
    """

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
