import asyncio
import json
import uuid

class MonitorClient():
    def __init__(self, socket_path = '/run/sparse/sparse-benchmark.sock'):
        self.socket_path = socket_path
        self.active_tasks = set()

    async def _send_message(self, message):
        try:
            reader, writer = await asyncio.open_unix_connection(self.socket_path)
            writer.write(message)
            writer.write_eof()
            writer.close()
        except Exception as e:
            pass

    def submit_event(self, task_payload):
        task = asyncio.create_task(self._send_message(json.dumps(task_payload).encode()))
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

        return task

    def start_benchmark(self, log_file_prefix = 'benchmark_sparse'):
        self.benchmark_id = str(uuid.uuid4())
        return self.submit_event({ "benchmark_id": self.benchmark_id,
                                   "event": "start",
                                   "log_file_prefix": log_file_prefix })

    def batch_processed(self, batch_size : int):
        return self.submit_event({ "benchmark_id": self.benchmark_id,
                                   "event": "batch_processed",
                                   "batch_size": batch_size })

    def task_processed(self):
        return self.submit_event({ "benchmark_id": self.benchmark_id,
                                   "event": "task_processed" })

