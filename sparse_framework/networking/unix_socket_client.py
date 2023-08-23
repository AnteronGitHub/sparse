import asyncio
import json

class UnixSocketClient:
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

