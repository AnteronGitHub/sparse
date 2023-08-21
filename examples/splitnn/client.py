import asyncio

from sparse_framework.dl import DatasetRepository, ModelMetaData

from utils import parse_arguments, _get_benchmark_log_file_prefix

from datasource import SplitNNDataSource
from gradient_calculator import GradientCalculator

class SplitNNClient(SplitNNDataSource):
    def __init__(self, **args):
        SplitNNDataSource.__init__(self, **args)

        self.task_executor = GradientCalculator(capacity = 1)
        self.task_executor.set_logger(self.logger)
        self.task_executor.set_node(self)
        self.task_executor.task_deployer = self.task_deployer
        self.task_executor.start()

    async def process_sample(self, features, labels):
        result_data = await self.task_executor.execute_task({ 'activation': features,
                                                              'labels': labels,
                                                              'model_meta_data': self.model_meta_data,
                                                              'capacity': 0 })

        return result_data['loss']

    async def loop_completed(self):
        await SplitNNDataSource.loop_completed(self)
        await self.task_executor.save_model(self.model_meta_data)

if __name__ == '__main__':
    args = parse_arguments()

    model_meta_data = ModelMetaData(1, args.model_name)
    dataset, classes = DatasetRepository().get_dataset(args.dataset)
    log_file_prefix = _get_benchmark_log_file_prefix(args, "datasource")

    splitnn_client = SplitNNClient(application=args.application,
                                   model_meta_data=model_meta_data,
                                   dataset=dataset,
                                   classes=classes)
    asyncio.run(splitnn_client.start(args.batch_size,
                                     args.batches,
                                     log_file_prefix=log_file_prefix,
                                     epochs=int(args.epochs)))
