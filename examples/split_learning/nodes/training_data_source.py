from torch.utils.data import DataLoader

from sparse.dl.serialization import encode_offload_request, decode_offload_response
from sparse.node.master import Master
from sparse.stats.monitor_server import MonitorClient

class TrainingDataSource(Master):
    def __init__(self, dataset, classes, model_name, benchmark = False):
        super().__init__()
        self.dataset = dataset
        self.classes = classes
        self.model_name = model_name
        self.benchmark = benchmark

    async def train(self, batch_size, batches, epochs, log_file_prefix):
        if self.benchmark:
            monitor_client = MonitorClient()
            await monitor_client.start()

        for t in range(epochs):
            for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                input_data = encode_offload_request(X, y)
                result_data = await self.task_deployer.deploy_task(input_data)
                split_grad, loss = decode_offload_response(result_data)

                if self.benchmark:
                    await monitor_client.batch_processed(len(X))

                if batch + 1 >= batches:
                    break

        if self.benchmark:
            await monitor_client.stop()
