import asyncio

from sparse_framework.dl import InferenceClient, DatasetRepository, ModelMetaData

from utils import parse_arguments, _get_benchmark_log_file_prefix

async def run_datasources(args):
    tasks = []
    dataset, classes = DatasetRepository().get_dataset(args.dataset)
    for i in range(args.no_datasources):
        datasource = InferenceClient(dataset,
                                     ModelMetaData(model_id=str(i % args.no_models), model_name=args.model_name),
                                     int(args.no_samples),
                                     int(args.use_scheduling)==1,
                                     node_id=str(i))
        tasks.append(datasource.start())

    await asyncio.gather(*tasks)

if __name__ == '__main__':
    args = parse_arguments()
    asyncio.run(run_datasources(args))

