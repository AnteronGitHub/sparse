import asyncio
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparse.dl.serialization import encode_offload_request, decode_offload_response
from sparse.node.master import Master
from sparse.stats.monitor_client import MonitorClient

class TrainingDataSource(Master):
    def __init__(self, dataset, classes, model_name, benchmark = True):
        super().__init__()
        self.dataset = dataset
        self.classes = classes
        self.model_name = model_name
        self.benchmark = benchmark

    async def train(self, batch_size, batches, epochs, log_file_prefix):
        if self.benchmark:
            monitor_client = MonitorClient()
            monitor_client.start_benchmark(log_file_prefix)

        progress_bar = tqdm(total=batch_size*batches*epochs,
                            unit='samples',
                            unit_scale=True)
        for t in range(epochs):
            for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                input_data = encode_offload_request(X, y)
                result_data = await self.task_deployer.deploy_task(input_data)
                split_grad, loss = decode_offload_response(result_data)

                progress_bar.update(len(X))
                if self.benchmark:
                    monitor_client.batch_processed(len(X))

                if batch + 1 >= batches:
                    break

        progress_bar.close()
        if self.benchmark:
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)

