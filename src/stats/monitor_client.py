import asyncio
import json

class MonitorClient():
    def __init__(self,
                 socket_path = 'sparse-benchmark.sock'):
        self.socket_path = socket_path
        self.active_tasks = set()

    async def _send_message(self, message):
        try:
            reader, writer = await asyncio.open_unix_connection(self.socket_path)
            print("Opened connection")
            writer.write(message)
            writer.write_eof()
            print("Waiting for th emessage to be sent")
            await writer.drain()
            await reader.read(100)
            writer.close()
            print("Closed writer")
        except Exception:
            pass

    def submit_event(self, task_payload):
        task = asyncio.create_task(self._send_message(json.dumps(task_payload).encode()))
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

    def start_benchmark(self, log_file_prefix = 'benchmark_sparse'):
        self.submit_event({"event": "start", "log_file_prefix": log_file_prefix})

    def stop_benchmark(self):
        self.submit_event({"event": "stop"})

    def batch_processed(self, batch_size : int):
        self.submit_event({"event": "batch_processed", "batch_size": batch_size})

    def task_processed(self):
        self.submit_event({"event": "task_processed"})

