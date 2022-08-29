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

    def start_benchmark(self, log_file_prefix = 'benchmark_sparse'):
        asyncio.create_task(self._send_message(json.dumps({"event": "start", "log_file_prefix": log_file_prefix}).encode()))

    def stop_benchmark(self):
        asyncio.create_task(self._send_message(json.dumps({"event": "stop"}).encode()))

    def batch_processed(self, batch_size : int):
        asyncio.create_task(self._send_message(json.dumps({"event": "batch_processed", "batch_size": batch_size}).encode()))

    def task_processed(self):
        asyncio.create_task(self._send_message(json.dumps({"event": "task_processed"}).encode()))

