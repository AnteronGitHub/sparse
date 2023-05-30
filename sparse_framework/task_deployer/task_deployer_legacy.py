import asyncio
import time

from ..task_deployer import TaskDeployer

class TaskDeployerLegacy(TaskDeployer):
    """Legacy asyncio implementation for older Python compiler versions.
    """

    async def stream_task_synchronous(self, input_data : bytes, loop):
        while True:
            try:
                task = asyncio.open_connection(self.upstream_host, self.upstream_port)
                reader, writer = await asyncio.wait_for(task, timeout=5)
                break
            except ConnectionRefusedError:
                self.logger.error("Unable to connect to upstream host. Trying again in 5 seconds...")
                time.sleep(5)
            except asyncio.exceptions.TimeoutError:
                self.logger.error("Connection to upstream host timed out. Retrying...")

        writer.write(input_data)
        writer.write_eof()

        # TODO: Better workaround is needed as opposed to ignoring...
        try:
            result_data = await reader.read()
            writer.close()

            return result_data
        except ConnectionResetError:
            self.logger.error("Connection reset by peer. Retrying...")
            return None

    async def deploy_task(self, input_data : bytes):
        loop = asyncio.get_event_loop()
        result = await asyncio.ensure_future(self.stream_task_synchronous(input_data, loop))
        # loop.close()
        return result
