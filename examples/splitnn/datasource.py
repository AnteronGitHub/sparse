import asyncio
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparse_framework import Master
from sparse_framework.dl import DatasetRepository

from utils import parse_arguments, _get_benchmark_log_file_prefix

class SplitNNDataSource(Master):
    def __init__(self, application, dataset, classes, benchmark = True):
        Master.__init__(self, benchmark=benchmark)
        self.application = application
        self.dataset = dataset
        self.classes = classes
        self.warmed_up = False

        self.progress_bar = None

    async def loop_starting(self, batch_size, batches, epochs, verbose):
        if verbose:
            self.progress_bar = tqdm(total=batch_size*batches*epochs,
                                     unit='samples',
                                     unit_scale=True)

    async def loop_completed(self):
        if self.progress_bar is not None:
            self.progress_bar.close()

        if self.monitor_client is not None:
            self.monitor_client.stop_benchmark()
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)

    async def task_completed(self, samples_processed, loss, log_file_prefix, processing_time : float):
        # Logging
        if self.progress_bar is not None:
            self.progress_bar.update(samples_processed)
        else:
            self.logger.info(f"Processed batch of {samples_processed} samples in {processing_time} seconds. Loss: {loss}")

        # Benchmarks
        if self.monitor_client is not None:
            if self.warmed_up:
                self.monitor_client.batch_processed(samples_processed, loss)
            else:
                self.warmed_up = True
                self.monitor_client.start_benchmark(log_file_prefix)

    async def process_sample(self, features, labels):
        result_data = await self.task_deployer.deploy_task({ 'activation': features, 'labels': labels })
        if self.is_learning():
            split_grad, loss = result_data['gradient'], result_data['loss']
        else:
            prediction = result_data['prediction']
            loss = None

        return loss

    async def start(self, batch_size, batches, epochs, log_file_prefix, verbose = False):
        await self.loop_starting(batch_size, batches, epochs, verbose)

        for t in range(epochs):
            offset = 0 if t == 0 else 1
            for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                task_started_at = time.time()
                loss = await self.process_sample(X, y)

                await self.task_completed(len(X), loss, log_file_prefix, processing_time=time.time() - task_started_at)

                if batch + offset >= batches:
                    break

        await self.loop_completed()

    def is_learning(self):
        return self.application == 'learning'

if __name__ == '__main__':
    args = parse_arguments()

    dataset, classes = DatasetRepository().get_dataset(args.dataset)
    log_file_prefix = _get_benchmark_log_file_prefix(args, "datasource")
    asyncio.run(SplitNNDataSource(args.application, dataset, classes).start(batch_size=args.batch_size,
                                                                            batches=args.batches,
                                                                            epochs=int(args.epochs),
                                                                            log_file_prefix=log_file_prefix))

