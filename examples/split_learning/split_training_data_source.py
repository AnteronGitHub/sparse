import asyncio

from sparse.node.master import Master
from sparse.dl.serialization import encode_offload_request, decode_offload_response

from datasets.cifar10 import load_CIFAR10_dataset
from training_benchmark import TrainingBenchmark

class SplitTrainingDataSource(Master):
    async def train(self):
        benchmark = TrainingBenchmark(model_name="VGG_unsplit")

        train_dataloader, classes = load_CIFAR10_dataset(benchmark.arguments.batch_size)
        benchmark.start()
        for t in range(benchmark.arguments.epochs):
            for batch, (X, y) in enumerate(train_dataloader):
                input_data = encode_offload_request(X, y)
                result_data = await self.task_deployer.deploy_task(input_data)
                split_grad, loss = decode_offload_response(result_data)

                benchmark.add_point(len(X))

                if batch + 1 >= benchmark.arguments.batches:
                    break

        benchmark.end()

if __name__ == "__main__":
    asyncio.run(SplitTrainingDataSource().train())
