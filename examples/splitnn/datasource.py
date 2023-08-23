import asyncio
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import uuid

from sparse_framework import Master
from sparse_framework.dl import DatasetRepository, ModelMetaData

from utils import parse_arguments, _get_benchmark_log_file_prefix

class SplitNNDataSource(Master):
    def __init__(self, application, model_meta_data, dataset, classes, data_source_id : str = str(uuid.uuid4()), benchmark = True):
        Master.__init__(self, benchmark=benchmark)
        self.application = application
        self.model_meta_data = model_meta_data
        self.dataset = dataset
        self.classes = classes

        self.id = data_source_id
        self.warmed_up = False

        self.progress_bar = None

    async def loop_starting(self, batches, epochs, verbose):
        self.logger.info(f"Starting data source {self.id}.")
        if verbose:
            self.progress_bar = tqdm(total=batches*epochs,
                                     unit='samples',
                                     unit_scale=True)

    async def loop_completed(self):
        self.logger.info(f"Stopping data source {self.id}.")
        if self.progress_bar is not None:
            self.progress_bar.close()

        if self.monitor_client is not None:
            self.monitor_client.stop_benchmark()
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)

    async def task_completed(self, loss, log_file_prefix, processing_time : float):
        # Logging
        if self.progress_bar is not None:
            self.progress_bar.update(1)
        else:
            self.logger.info(f"{self.id}: Processed a sample in {processing_time} seconds. Loss: {loss}")

        # Benchmarks
        if self.monitor_client is not None:
            if self.warmed_up:
                self.monitor_client.task_completed(processing_time)
            else:
                self.warmed_up = True
                self.monitor_client.start_benchmark(f"{log_file_prefix}-monitor")
                self.monitor_client.start_benchmark(f"{log_file_prefix}-tasks", benchmark_type="ClientBenchmark")

    async def process_sample(self, features, labels):
        result_data = await self.task_deployer.deploy_task({ 'activation': features,
                                                             'labels': labels,
                                                             'model_meta_data': self.model_meta_data,
                                                             'capacity': 0 })
        if self.is_learning():
            split_grad, loss = result_data['gradient'], result_data['loss']
        else:
            prediction = result_data['prediction']
            loss = None

        return loss

    async def start(self, batches, epochs, log_file_prefix, verbose = False, time_window = 5):
        await self.loop_starting(batches, epochs, verbose)

        for t in range(epochs):
            offset = 0 if t == 0 else 1
            for batch, (X, y) in enumerate(DataLoader(self.dataset, 1)):
                task_started_at = time.time()
                loss = await self.process_sample(X, y)

                processing_time = time.time() - task_started_at
                await self.task_completed(loss, log_file_prefix, processing_time=processing_time)
                await asyncio.sleep(time_window - processing_time)

                if batch + offset >= batches:
                    break

        await self.loop_completed()

    def is_learning(self):
        return self.application == 'learning'

async def run_datasources(args):
    tasks = []
    dataset, classes = DatasetRepository().get_dataset(args.dataset)
    for i in range(args.no_datasources):
        model_id = str(i % args.no_models)
        model_meta_data = ModelMetaData(model_id, args.model_name)
        log_file_prefix = _get_benchmark_log_file_prefix(args, f"datasource{i}")
        datasource = SplitNNDataSource(args.application,
                                       model_meta_data,
                                       dataset,
                                       classes,
                                       data_source_id=f"datasource{i}")
        tasks.append(datasource.delay_coro(datasource.start,
                                           args.batches,
                                           int(args.epochs),
                                           log_file_prefix,
                                           delay=i))

    await asyncio.gather(*tasks)

if __name__ == '__main__':
    args = parse_arguments()
    asyncio.run(run_datasources(args))

