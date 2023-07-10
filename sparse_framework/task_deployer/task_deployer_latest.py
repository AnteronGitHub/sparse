import asyncio

from .task_deployer_base import TaskDeployerBase

class TaskDeployerLatest(TaskDeployerBase):
    """Task deployer implementation using the current asyncio interface.
    """

    async def stream_task_synchronous(self, input_data : bytes):
        while True:
            try:
                task = asyncio.open_connection(self.node.config_manager.upstream_host, self.node.config_manager.upstream_port)
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

        # TODO: Better workaround is needed as opposed to retrying...
        try:
            result_data = await reader.read()
            writer.close()

            return result_data
        except ConnectionResetError:
            self.logger.error("Connection reset by peer. Retrying...")
            return None

    async def deploy_task(self, input_data : dict):
        payload = self.encode_request(input_data)
        while True:
            task = asyncio.create_task(self.stream_task_synchronous(payload))
            await task
            if task.result() is None:
                self.logger.error(f"Broken pipe error. Re-executing...")
                if self.node.monitor_client is not None:
                    self.node.monitor_client.broken_pipe_error()
            else:
                task_result = self.decode_response(task.result())
                return task_result
