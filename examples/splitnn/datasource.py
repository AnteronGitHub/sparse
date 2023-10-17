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

class ModelClientProtocol(asyncio.Protocol):
    def __init__(self, data_source_id, dataset, model_meta_data, on_con_lost, no_samples = 64):
        self.dataset = dataset
        self.model_meta_data = model_meta_data
        self.on_con_lost = on_con_lost
        self.no_samples = no_samples

        self.logger = logging.getLogger(f"sparse datasource {data_source_id}")
        self.processing_started = None
        self.last_sent_at = None
        self.latencies = []
        self.request_latencies = []

    def print_statistics(self):
        no_requests = len(self.latencies)
        if no_requests == 0:
            self.logger.info(f"Connection closed || No requests made during connection.")
        else:
            avg_latency = sum(self.latencies)/len(self.latencies)
            avg_request_latency = sum(self.request_latencies)/len(self.request_latencies)
            self.logger.info(f"Connection closed, {no_requests} requests served, statistics: avg E2E lat. / avg Req lat.: {1000*avg_latency:.2f} ms / {1000*avg_request_latency:.2f} ms.")

    def process_sample(self):
        self.processing_started = time()
        for batch, (X, y) in enumerate(DataLoader(self.dataset, 1)):
            self.logger.debug("Sending data")
            self.transport.write(pickle.dumps({ 'activation': X,
                'labels': y,
                'model_meta_data': self.model_meta_data }))
            self.last_sent_at = time()
            break

    def connection_made(self, transport):
        self.logger.info("Connected to downstream host.")
        self.transport = transport
        self.process_sample()

    def data_received(self, data):
        latency = time() - self.processing_started
        request_latency = time() - self.last_sent_at
        self.latencies.append(latency)
        self.request_latencies.append(request_latency)
        if (self.no_samples > 0):
            self.process_sample()
            self.no_samples -= 1
        else:
            self.transport.close()

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
        model_meta_data = ModelMetaData(model_id, args.model_name)
        log_file_prefix = _get_benchmark_log_file_prefix(args, f"datasource{i}")
        datasource = SplitNNDataSource(str(i))
        tasks.append(datasource.delay_coro(datasource.start,
                                           dataset,
                                           model_meta_data,
                                           args.batches*int(args.epochs),
                                           delay=0))

    await asyncio.gather(*tasks)

if __name__ == '__main__':
    args = parse_arguments()
    asyncio.run(run_datasources(args))

