import asyncio
from sparse_framework import SparseNode

from datasets import get_dataset
from model_meta_data import ModelMetaData
from protocols import InferenceClientProtocol
from utils import parse_arguments, _get_benchmark_log_file_prefix

class InferenceClient(SparseNode):
    """A Node that iterates over a dataset and offloads the sample inference to specified server.
    """
    def __init__(self, dataset, model_meta_data, no_samples, use_scheduling, target_latency, **kwargs):
        super().__init__(**kwargs)
        self.protocol_factory = lambda on_con_lost, stats_queue: \
                                        lambda: InferenceClientProtocol(self.node_id, \
                                                                        dataset, \
                                                                        model_meta_data, \
                                                                        on_con_lost, \
                                                                        no_samples, \
                                                                        use_scheduling, \
                                                                        target_latency, \
                                                                        stats_queue=stats_queue)


    def get_futures(self):
        futures = super().get_futures()

        futures.append(self.connect_to_server(self.protocol_factory,
                                              self.config.upstream_host,
                                              self.config.upstream_port,
                                              self.result_callback))

        return futures

    def result_callback(self, result):
        self.logger.info(result)


async def run_datasources(args):
    tasks = []
    dataset, classes = get_dataset(args.dataset)
    for i in range(args.no_datasources):
        datasource = InferenceClient(dataset,
                                     ModelMetaData(model_id=str(i % args.no_models), model_name=args.model_name),
                                     int(args.no_samples),
                                     int(args.use_scheduling)==1,
                                     float(args.target_latency)/1000.0,
                                     node_id=str(i))
        tasks.append(datasource.start())

    await asyncio.gather(*tasks)

if __name__ == '__main__':
    args = parse_arguments()
    asyncio.run(run_datasources(args))

