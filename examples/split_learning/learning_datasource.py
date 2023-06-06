import asyncio
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparse_framework.dl.serialization import encode_offload_request, decode_offload_response
from sparse_framework.dl.serialization import encode_offload_request_pruned
from sparse_framework.node.master import Master

from datasets import DatasetRepository
from utils import parse_arguments, get_depruneProps, get_deprune_epochs, _get_benchmark_log_file_prefix

class LearningDataSource(Master):
    def __init__(self, dataset, classes, benchmark = True):
        super().__init__(benchmark=benchmark)
        self.dataset = dataset
        self.classes = classes
        self.warmed_up = False

    async def train(self, batch_size, batches, depruneProps, log_file_prefix, use_compression, epochs, verbose = False):
        total_ephochs = get_deprune_epochs(depruneProps)
        if verbose:
            progress_bar = tqdm(total=batch_size*batches*total_ephochs,
                                unit='samples',
                                unit_scale=True)
        else:
            progress_bar = None

        if use_compression:
            phases = depruneProps
        else:
            phases = [{'epochs': epochs, 'budget': None}]

        for phase, prop in enumerate(phases):
            epochs = prop['epochs']
            budget = prop['budget']
            for t in range(epochs):
                offset = 0 if phase == 0 and t == 0 else 1 # Ensures that an extra batch is processed in the first epoch since one batch is for warming up
                for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                    if use_compression:
                        input_data = encode_offload_request_pruned(X, y, None, budget)
                    else:
                        input_data = encode_offload_request(X, y)

                    while True:
                        result_data = await self.task_deployer.deploy_task(input_data)
                        if result_data is None:
                            self.logger.error(f"Broken pipe error. Re-executing...")
                            if self.monitor_client is not None:
                                self.monitor_client.broken_pipe_error()
                        else:
                            break
                    split_grad, loss = decode_offload_response(result_data)

                    # Logging
                    if progress_bar is not None:
                        progress_bar.update(len(X))
                    else:
                        self.logger.info(f"Processed batch of {len(X)} samples")

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
    depruneProps = get_depruneProps(args)
    log_file_prefix = _get_benchmark_log_file_prefix(args, "datasource", get_deprune_epochs(depruneProps))
    asyncio.run(LearningDataSource(dataset, classes).train(args.batch_size,
                                                           args.batches,
                                                           depruneProps,
                                                           log_file_prefix=log_file_prefix,
                                                           use_compression=bool(args.use_compression),
                                                           epochs=int(args.epochs)))

