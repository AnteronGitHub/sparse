import asyncio

from ..file_logger import FileLogger

class BaseBenchmark:
    """Common base class for benchmarks.

    Each benchmark instance logs results to its own file.
    """
    def __init__(self, benchmark_id, log_file_prefix, stop_callback, logger, timeout : int):
        self.benchmark_id = benchmark_id
        self.stats_logger = FileLogger(benchmark_id, file_prefix=log_file_prefix)
        self.stop_callback = stop_callback
        self.logger = logger
        self.timeout = timeout
        self.timeout_task = None

    def write_log(self, stats : list):
        self.stats_logger.log_row(stats)

    async def timeout_stop(self):
        await asyncio.sleep(self.timeout)
        self.stop()

    def start(self):
        self.timeout_task = asyncio.create_task(self.timeout_stop())

    def receive_message(self, payload):
        self.timeout_task.cancel()
        self.timeout_task = asyncio.create_task(self.timeout_stop())

    def stop(self):
        self.timeout_task.cancel()
        self.stop_callback(self)
        self.logger.info(f"Stopped benchmark '{self.benchmark_id}'")

