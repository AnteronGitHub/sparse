import asyncio

from .base_tcp_client import BaseTCPClient

class TCPClientLegacy(BaseTCPClient):
    """Legacy asyncio TCP client implementation for older Python compiler versions.
    """

    async def _create_request(self, request_data : bytes):
        payload = self.encode_data(request_data)

        # TODO: Better workaround is needed as opposed to retrying...
        while True:
            loop = asyncio.get_event_loop()
            result = await asyncio.ensure_future(self._send_request(payload, loop))
            if task.result() is None:
                self.broken_pipe_error()
            else:
                task_result = self.decode_data(task.result())
                return task_result
