import asyncio
from tqdm import tqdm
from torch.utils.data import DataLoader

from sparse_framework.dl.serialization import encode_offload_inference_request, encode_offload_inference_request_pruned
from sparse_framework.node.master import Master
from sparse_framework.stats.monitor_client import MonitorClient

from datasets.yolov3 import YOLOv3Dataset
import torch

from benchmark import parse_arguments, get_depruneProps, _get_benchmark_log_file_prefix

class InferenceDataSourceYOLO(Master):
    def __init__(self, benchmark = True):
        super().__init__()
        self.dataset = YOLOv3Dataset()
        if benchmark:
            self.monitor_client = MonitorClient()
        else:
            self.monitor_client = None

    async def start(self, inferences_to_be_run = 100, img_size=416, log_file_prefix = None, verbose = False):
        progress_bar = tqdm(total=inferences_to_be_run,
                            unit='inferences',
                            unit_scale=True)
        if self.monitor_client is not None:
            self.monitor_client.start_benchmark(log_file_prefix)
        for t in range(inferences_to_be_run):
            X = self.dataset.get_sample(img_size).to('cpu')

            result_data = await self.task_deployer.deploy_task(encode_offload_inference_request(X))
            if self.monitor_client is not None:
                self.monitor_client.batch_processed(len(X))
            progress_bar.update(1)

        progress_bar.close()
        if self.monitor_client is not None:
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)


class InferenceDataSource(Master):
    def __init__(self, dataset, benchmark = True):
        Master.__init__(self, benchmark=benchmark)
        self.dataset = dataset

    async def start(self, batch_size, batches, depruneProps, use_compression, epochs, log_file_prefix, verbose = False):
        if self.monitor_client is not None:
            self.monitor_client.start_benchmark(log_file_prefix)

        if verbose:
            inferences_to_be_run = batch_size * batches
            progress_bar = tqdm(total=inferences_to_be_run,
                                unit='inferences',
                                unit_scale=True)
        else:
            progress_bar = None

        if use_compression:
            phases = depruneProps
        else:
            phases = [{'epochs': epochs, 'budget': None, 'pruneState': 0}]

        for phase, prop in enumerate(phases):
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

    from datasets import DatasetRepository
    dataset, classes = DatasetRepository().get_dataset(args.dataset)
    depruneProps = get_depruneProps(args)
    use_compression = bool(args.use_compression)

    asyncio.run(InferenceDataSource(dataset).start(args.batch_size,
                                                   args.batches,
                                                   depruneProps,
                                                   use_compression,
                                                   epochs=int(args.epochs),
                                                   log_file_prefix=_get_benchmark_log_file_prefix(args)))
