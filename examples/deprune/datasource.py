import asyncio
from tqdm import tqdm
from torch.utils.data import DataLoader

from sparse_framework.node.master import Master
from sparse_framework.dl import DatasetRepository
from sparse_framework.stats.monitor_client import MonitorClient


from serialization import encode_offload_inference_request, encode_offload_inference_request_pruned
from benchmark import parse_arguments, get_depruneProps, _get_benchmark_log_file_prefix

class DepruneDataSource(Master):
    def __init__(self, dataset, benchmark = True):
        Master.__init__(self, benchmark=benchmark)
        self.dataset = dataset

    async def start(self, batch_size, batches, depruneProps, log_file_prefix, verbose = False):
        if self.monitor_client is not None:
            self.monitor_client.start_benchmark(log_file_prefix)

        if verbose:
            inferences_to_be_run = batch_size * batches
            progress_bar = tqdm(total=inferences_to_be_run,
                                unit='inferences',
                                unit_scale=True)
        else:
            progress_bar = None

        for phase, prop in enumerate(depruneProps):
            epochs = prop['epochs']
            pruneState = prop['pruneState']
            budget = prop['budget']
            for t in range(epochs):
                for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                    if pruneState:
                        input_data = encode_offload_inference_request_pruned(X, None, budget)
                    else:
                        input_data = encode_offload_inference_request(X)

                    result_data = await self.task_deployer.deploy_task(input_data)

                    if self.monitor_client is not None:
                        self.monitor_client.batch_processed(len(X))
                    if progress_bar is not None:
                        progress_bar.update(len(X))
                    else:
                        self.logger.info(f"Processed batch of {len(X)} samples")

                    if batch + 1 >= batches:
                        break

        progress_bar.close()
        if self.monitor_client is not None:
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)

if __name__ == "__main__":
    args = parse_arguments()

    dataset, classes = DatasetRepository().get_dataset(args.dataset)
    depruneProps = get_depruneProps(args)

    asyncio.run(DepruneDataSource(dataset).start(args.batch_size,
                                                 args.batches,
                                                 depruneProps,
                                                 log_file_prefix=_get_benchmark_log_file_prefix(args)))
