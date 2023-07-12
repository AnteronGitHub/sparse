import asyncio

from .base_tcp_client import BaseTCPClient

class TCPClientLatest(BaseTCPClient):
    """TCP client implementation using the current asyncio interface.
    """

    async def _create_request(self, request_data : dict) -> dict:
        payload = self.encode_data(request_data)

        # TODO: Better workaround is needed as opposed to retrying...
        while True:
            task = asyncio.create_task(self._send_request(payload))
            await task
            if task.result() is None:
                self.broken_pipe_error()
            else:
                task_result = self.decode_data(task.result())
                return task_result
