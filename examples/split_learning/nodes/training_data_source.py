from torch.utils.data import DataLoader

from sparse.dl.serialization import encode_offload_request, decode_offload_response
from sparse.node.master import Master
from sparse.stats.training_benchmark import TrainingBenchmark

class TrainingDataSource(Master):
    def __init__(self, dataset, classes, model_name):
        super().__init__()
        self.dataset = dataset
        self.classes = classes
        self.model_name = model_name

    async def train(self, batch_size, batches, epochs, log_file_prefix):
        benchmark = TrainingBenchmark(model_name=self.model_name,
                                      batches=batches,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      log_file_prefix=log_file_prefix)

        benchmark.start()
        for t in range(epochs):
            for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                input_data = encode_offload_request(X, y)
                result_data = await self.task_deployer.deploy_task(input_data)
                split_grad, loss = decode_offload_response(result_data)

                benchmark.add_point(len(X))

                if batch + 1 >= batches:
                    break

        benchmark.end()
