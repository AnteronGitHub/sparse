import asyncio
import json

class MonitorClient():
    def __init__(self,
                 socket_path = 'sparse-benchmark.socket'):
        self.socket_path = socket_path
        self.active_tasks = set()

    async def _send_message(self, message):
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        writer.write(message)
        writer.write_eof()
        data = await reader.read(100)
        writer.close()

    def start_benchmark(self, log_file_prefix = 'benchmark_sparse'):
        task = asyncio.create_task(self._send_message(json.dumps({"event": "start", "log_file_prefix": log_file_prefix}).encode()))
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

    def stop_benchmark(self):
        task = asyncio.create_task(self._send_message(json.dumps({"event": "stop"}).encode()))
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

    def batch_processed(self, batch_size : int):
        task = asyncio.create_task(self._send_message(json.dumps({"event": "batch_processed", "batch_size": batch_size}).encode()))
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

    def task_processed(self):
        task = asyncio.create_task(self._send_message(json.dumps({"event": "task_processed"}).encode()))
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

