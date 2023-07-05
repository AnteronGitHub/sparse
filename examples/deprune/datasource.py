import asyncio
from tqdm import tqdm
from torch.utils.data import DataLoader

from sparse_framework import Master, MonitorClient
from sparse_framework.dl import DatasetRepository

from serialization import encode_offload_request, \
                          encode_offload_request_pruned, \
                          encode_offload_inference_request, \
                          encode_offload_inference_request_pruned
from benchmark import parse_arguments, get_depruneProps, _get_benchmark_log_file_prefix

class DepruneDataSource(Master):
    def __init__(self, application, dataset, benchmark = True):
        Master.__init__(self, benchmark=benchmark)
        self.application = application
        self.dataset = dataset
        self.warmed_up = False

    def is_learning(self):
        return self.application == 'learning'

    def encode_request(self, X, y, pruneState, budget):
        if pruneState:
            if self.is_learning():
                return encode_offload_request_pruned(X, y, None, budget)
            else:
                return encode_offload_inference_request_pruned(X, None, budget)
        else:
            if self.is_learning():
                return encode_offload_request(X, y)
            else:
                return encode_offload_inference_request(X)

    async def start(self, batch_size, batches, depruneProps, log_file_prefix, verbose = False):
        if self.monitor_client is not None:
            self.monitor_client.start_benchmark(log_file_prefix)

        if verbose:
            inferences_to_be_run = batch_size * batches
            progress_bar = tqdm(total=inferences_to_be_run,
                                unit='samples',
                                unit_scale=True)
        else:
            progress_bar = None

        for phase, prop in enumerate(depruneProps):
            epochs = prop['epochs']
            pruneState = prop['pruneState']
            budget = prop['budget']
            for t in range(epochs):
                for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                    input_data = self.encode_request(X, y, pruneState, budget)

                    result_data = await self.task_deployer.deploy_task(input_data)

                    # Benchmarks
                    if self.monitor_client is not None:
                        if self.warmed_up:
                            self.monitor_client.batch_processed(len(X), loss)
                        else:
                            self.warmed_up = True
                            self.monitor_client.start_benchmark(log_file_prefix)

                    # Logging
                    if progress_bar is not None:
                        progress_bar.update(len(X))
                    else:
                        self.logger.info(f"Processed batch of {len(X)} samples")

                    if batch + 1 >= batches:
                        break

        if progress_bar is not None:
            progress_bar.close()
        if self.monitor_client is not None:
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)

if __name__ == "__main__":
    args = parse_arguments()

    dataset, classes = DatasetRepository().get_dataset(args.dataset)
    depruneProps = get_depruneProps(args)

    asyncio.run(DepruneDataSource(application=args.application,
                                  dataset=dataset).start(args.batch_size,
                                                         args.batches,
                                                         depruneProps,
                                                         log_file_prefix=_get_benchmark_log_file_prefix(args, "datasource")))
