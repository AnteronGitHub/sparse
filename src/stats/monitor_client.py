import asyncio
import json

class MonitorClient():
    def __init__(self,
                 socket_path = 'sparse-benchmark.socket'):
        self.socket_path = socket_path

    async def _send_message(self, message):
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        writer.write(message)
        writer.write_eof()
        data = await reader.read(100)
        writer.close()

    async def start_benchmark(self):
        await self._send_message(json.dumps({"event": "start"}).encode())

    async def stop_benchmark(self):
        await self._send_message(json.dumps({"event": "stop"}).encode())

    async def batch_processed(self, batch_size : int):
        await self._send_message(json.dumps({"event": "batch_processed", "batch_size": batch_size}).encode())

    async def task_processed(self):
        await self._send_message(json.dumps({"event": "task_processed"}).encode())

