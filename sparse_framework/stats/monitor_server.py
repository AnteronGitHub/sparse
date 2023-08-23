import asyncio
import logging
import os

from sparse_framework.daemons import ClockedLoop
from sparse_framework.networking import UnixSocketServer

from .benchmark import ClientBenchmark, MonitorBenchmark
from .monitor import NodeMonitor

class MonitorServer(UnixSocketServer, ClockedLoop):
    def __init__(self,
                 update_frequency_ps = 8,
                 socket_path = '/run/sparse/sparse-benchmark.sock',
                 benchmark_timeout = 30):
        UnixSocketServer.__init__(self, socket_path)
        ClockedLoop.__init__(self, update_frequency_ps)

        self.benchmark_timeout = benchmark_timeout

        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=logging.INFO)
        self.logger = logging.getLogger("sparse")
        self.nic = os.environ.get('SPARSE_MONITOR_NIC') or ''

        nic_name = self.nic or 'all'
        self.logger.info(f"Monitoring NIC '{nic_name}'")

        self.client_benchmarks = set()
        self.monitor_benchmarks = set()

    def loop_task(self):
        for benchmark in self.monitor_benchmarks:
            benchmark.log_stats()

    def start_benchmark(self, request_data : dict):
        if request_data['benchmark_type'] == 'ClientBenchmark':
            benchmark = ClientBenchmark(request_data['benchmark_id'],
                                        request_data['log_file_prefix'],
                                        self.client_benchmarks.discard,
                                        self.logger,
                                        self.benchmark_timeout)
            self.client_benchmarks.add(benchmark)
            benchmark.start()
        elif request_data['benchmark_type'] == 'MonitorBenchmark':
            benchmark = MonitorBenchmark(request_data['benchmark_id'],
                                         request_data['log_file_prefix'],
                                         self.monitor_benchmarks.discard,
                                         self.logger,
                                         self.benchmark_timeout,
                                         monitor_container=NodeMonitor(nic=self.nic))
            self.monitor_benchmarks.add(benchmark)
            benchmark.start()
        self.logger.info(f"Started benchmark '{request_data['benchmark_id']}' with log prefix '{request_data['log_file_prefix']}'.")

    def handle_request(self, request_data : dict) -> None:
        if request_data['event'] == 'start':
            try:
                self.start_benchmark(request_data)
            except Exception as e:
                self.logger.error(f"Unable to start benchmark from message {request_data}")
                self.logger.error(e)
        else:
            for benchmark in self.client_benchmarks:
                if benchmark.benchmark_id == request_data['benchmark_id']:
                    benchmark.receive_message(request_data)
                    return

            for benchmark in self.monitor_benchmarks:
                if benchmark.benchmark_id == request_data['benchmark_id']:
                    benchmark.receive_message(request_data)
                    return

    async def start(self):
        await asyncio.gather(self.run_unix_server(), self.run_clocked_loop())
