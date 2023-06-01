import asyncio

from ..task_deployer import TaskDeployer

class TaskDeployerLatest(TaskDeployer):
    """Task deployer implementation using the current asyncio interface.
    """

    async def stream_task_synchronous(self, input_data : bytes):
        while True:
            try:
                task = asyncio.open_connection(self.upstream_host, self.upstream_port)
                reader, writer = await asyncio.wait_for(task, timeout=5)
                break
            except ConnectionRefusedError:
                self.logger.error("Unable to connect to upstream host. Trying again in 5 seconds...")
                await asyncio.sleep(5)
            except asyncio.exceptions.TimeoutError:
                self.logger.error("Connection to upstream host timed out. Retrying...")
                if self.node.monitor_client is not None:
                    self.node.monitor_client.connection_timeout()

        writer.write(input_data)
        writer.write_eof()
        await writer.drain()

        # TODO: Better workaround is needed as opposed to ignoring...
        try:
            result_data = await reader.read()
            writer.close()

            return result_data
        except ConnectionResetError:
            self.logger.error("Connection reset by peer. Retrying...")
            return None

    async def deploy_task(self, input_data : bytes):
        task = asyncio.create_task(self.stream_task_synchronous(input_data))
        await task
        return task.result()
