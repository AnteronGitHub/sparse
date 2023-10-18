import asyncio
import logging
from time import time
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import uuid

from sparse_framework import Master
from sparse_framework.dl import DatasetRepository, ModelMetaData

from utils import parse_arguments, _get_benchmark_log_file_prefix

TARGET_FPS = 5.0

class ModelClientProtocol(asyncio.Protocol):
    def __init__(self, data_source_id, dataset, model_meta_data, on_con_lost, no_samples = 64, target_rate = 1/TARGET_FPS):
        self.dataloader = DataLoader(dataset, 1)
        self.model_meta_data = model_meta_data
        self.on_con_lost = on_con_lost
        self.no_samples = no_samples
        self.target_rate = target_rate

        self.logger = logging.getLogger(f"sparse datasource {data_source_id}")
        self.processing_started = None
        self.last_sent_at = None

        self.latencies = []
        self.request_latencies = []
        self.ratios = []

    def initialize_stream(self):
        self.processing_started = time()
        self.transport.write(pickle.dumps({ 'op': "initialize_stream", 'model_meta_data': self.model_meta_data }))
        self.last_sent_at = time()

    def stream_initialized(self, result_data):
        latency = time() - self.processing_started
        self.logger.info(f"Initialized stream in {latency:.2f} seconds with {1.0/self.target_rate:.2f} FPS target rate.")
        self.offload_task()

    def offload_task(self):
        self.processing_started = time()
        self.no_samples -= 1
        features, labels = next(iter(self.dataloader))
        self.transport.write(pickle.dumps({ 'op': 'offload_task',
                                            'activation': features,
                                            'labels': labels,
                                            'model_meta_data': self.model_meta_data }))
        self.last_sent_at = time()

    def offload_task_completed(self, result_data):
        latency = time() - self.processing_started
        request_latency = time() - self.last_sent_at

        self.latencies.append(latency)
        self.request_latencies.append(request_latency)
        self.ratios.append(request_latency/latency)

        if (self.no_samples > 0):
            loop = asyncio.get_running_loop()
            loop.call_later( self.target_rate-latency if self.target_rate-latency > 0 else 0, self.offload_task)
        else:
            self.transport.close()

    def print_statistics(self):
        no_requests = len(self.latencies)
        if no_requests == 0:
            self.logger.info(f"Connection closed || No requests made during connection.")
        else:
            avg_latency = sum(self.latencies)/len(self.latencies)
            avg_request_latency = sum(self.request_latencies)/len(self.request_latencies)
            avg_ratio = sum(self.ratios)/len(self.ratios)
            self.logger.info(f"Stream statistics: {no_requests} tasks / {1.0/avg_latency:.2f} avg FPS / {1000*avg_request_latency:.2f} ms avg request latency / {100.0*avg_ratio:.2f} % avg offload latency ratio.")

    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        self.logger.info(f"Connected to downstream host on '{peername}'.")
        self.transport = transport
        self.initialize_stream()

    def data_received(self, data):
        result_data = pickle.loads(data)

        if "statusCode" in result_data.keys():
            self.stream_initialized(result_data)
        else:
            self.offload_task_completed(result_data)

    def connection_lost(self, exc):
        self.print_statistics()
        self.on_con_lost.set_result(True)

class SplitNNDataSource(Master):
    def __init__(self, data_source_id : str = str(uuid.uuid4()), benchmark = True):
        Master.__init__(self, benchmark=benchmark)
        self.id = data_source_id

    async def start(self, dataset, model_meta_data, no_samples):
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()
        transport, protocol = await loop.create_connection(lambda: ModelClientProtocol(self.id,
                                                                                       dataset,
                                                                                       model_meta_data,
                                                                                       on_con_lost),
                                                           self.config_manager.upstream_host,
                                                           self.config_manager.upstream_port)
        await on_con_lost

async def run_datasources(args):
    tasks = []
    dataset, classes = DatasetRepository().get_dataset(args.dataset)
    for i in range(args.no_datasources):
        model_id = str(i % args.no_models)
        datasource = SplitNNDataSource(str(i))
        tasks.append(datasource.delay_coro(datasource.start,
                                           dataset,
                                           ModelMetaData(model_id, args.model_name),
                                           args.batches*int(args.epochs),
                                           delay=0))

    await asyncio.gather(*tasks)

if __name__ == '__main__':
    args = parse_arguments()
    asyncio.run(run_datasources(args))

