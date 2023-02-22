import asyncio
import json
import logging
import socket
import time

from .benchmark import Benchmark

class MonitorServer():
    def __init__(self,
                 update_frequency_ps = 8,
                 socket_path = '/run/sparse/sparse-benchmark.sock'):

        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=logging.INFO)
        self.logger = logging.getLogger("sparse")
        self.update_frequency_ps = update_frequency_ps
        self.socket_path = socket_path

        self.benchmarks = set()
        self.stopped_benchmarks = set()

    def log_stats(self):
        for benchmark in self.benchmarks:
            benchmark.log_stats()
        for benchmark in self.stopped_benchmarks:
            self.benchmarks.discard(benchmark)
            self.logger.info(f"Stopped benchmark '{benchmark.benchmark_id}'")
        self.stopped_benchmarks.clear()

    def start_benchmark(self, benchmark_id, log_file_prefix):
        self.benchmarks.add(Benchmark(benchmark_id,
                                      log_file_prefix,
                                      self.stopped_benchmarks.add))
        self.logger.info(f"Started a new benchmark '{benchmark_id}' with log prefix '{log_file_prefix}'")

    async def run_monitor(self):
        self.logger.info("Starting monitor")
        while True:
            start_time = time.time()
            self.log_stats()
            time_elapsed = time.time() - start_time
            await asyncio.sleep(1.0/self.update_frequency_ps - time_elapsed)

    async def receive_message(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        input_data = await reader.read()
        writer.write("ACK".encode())
        writer.write_eof()
        writer.close()

        payload = json.loads(input_data.decode())
        for benchmark in self.benchmarks:
            if benchmark.benchmark_id == payload['benchmark_id']:
                benchmark.receive_message(payload)
                return
        self.start_benchmark(payload['benchmark_id'], payload['log_file_prefix'])

    async def run_server(self):
        self.logger.info(f"Starting the monitoring server on '{self.socket_path}'")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server = await asyncio.start_unix_server(self.receive_message, path=self.socket_path)
        await server.serve_forever()

    async def run(self):
        await asyncio.gather(self.run_server(), self.run_monitor())

    def start(self):
        asyncio.run(self.run())

