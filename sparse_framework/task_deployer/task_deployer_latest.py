import asyncio
import time

from ..task_deployer import TaskDeployer

class TaskDeployerLatest(TaskDeployer):
    """Task deployer implementation using the current asyncio interface.
    """

    async def stream_task_synchronous(self, input_data : bytes):
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.upstream_host, self.upstream_port)
                break
            except ConnectionRefusedError:
                self.logger.error("Unable to connect to upstream host. Trying again in 5 seconds...")
                time.sleep(5)
            except TimeoutError:
                self.logger.error("Connection to upstream host timed out. Retrying...")

        writer.write(input_data)
        writer.write_eof()
        await writer.drain()

        result_data = await reader.read()
        writer.close()

        return result_data

    async def deploy_task(self, input_data : bytes):
        task = asyncio.create_task(self.stream_task_synchronous(input_data))
        await task
        return task.result()
