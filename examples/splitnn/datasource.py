import asyncio
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparse_framework import Master
from sparse_framework.dl import DatasetRepository

from utils import parse_arguments, _get_benchmark_log_file_prefix

class SplitNNDataSource(Master):
    def __init__(self, application, dataset, classes, benchmark = True):
        super().__init__(benchmark=benchmark)
        self.application = application
        self.dataset = dataset
        self.classes = classes
        self.warmed_up = False

    def is_learning(self):
        return self.application == 'learning'

    async def start(self, batch_size, batches, epochs, log_file_prefix, verbose = False):
        if verbose:
            progress_bar = tqdm(total=batch_size*batches*epochs,
                                unit='samples',
                                unit_scale=True)
        else:
            progress_bar = None

        for t in range(epochs):
            offset = 0 if t == 0 else 1
            for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):

                while True:
                    result_data = await self.task_deployer.deploy_task({ 'activation': X, 'labels': y })
                    if result_data is None:
                        self.logger.error(f"Broken pipe error. Re-executing...")
                        if self.monitor_client is not None:
                            self.monitor_client.broken_pipe_error()
                    else:
                        break
                if self.is_learning():
                    split_grad, loss = result_data['gradient'], result_data['loss']
                else:
                    prediction = result_data['prediction']
                    loss = None

                # Logging
                if progress_bar is not None:
                    progress_bar.update(len(X))
                else:
                    self.logger.info(f"Processed batch of {len(X)} samples. Loss: {loss}")

                # Benchmarks
                if self.monitor_client is not None:
                    if self.warmed_up:
                        self.monitor_client.batch_processed(len(X), loss)
                    else:
                        self.warmed_up = True
                        self.monitor_client.start_benchmark(log_file_prefix)

                if batch + offset >= batches:
                    break

        if progress_bar is not None:
            progress_bar.close()

        if self.monitor_client is not None:
            self.monitor_client.stop_benchmark()
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)

if __name__ == '__main__':
    args = parse_arguments()

    dataset, classes = DatasetRepository().get_dataset(args.dataset)
    log_file_prefix = _get_benchmark_log_file_prefix(args, "datasource")
    asyncio.run(SplitNNDataSource(args.application, dataset, classes).start(batch_size=args.batch_size,
                                                                            batches=args.batches,
                                                                            epochs=int(args.epochs),
                                                                            log_file_prefix=log_file_prefix))

