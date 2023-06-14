from sparse_framework.node.worker import Worker
from sparse_framework.dl.inference_calculator import InferenceCalculator

from benchmark import parse_arguments, get_depruneProps

class InferenceWorker(Worker):
    def __init__(self, model_name, partition, compressionProps, use_compression):
        if use_compression:
            from sparse_framework.dl import InferenceCalculatorPruning
            task_executor = InferenceCalculatorPruning(model_name, partition, compressionProps, use_compression)
        else:
            from sparse_framework.dl import InferenceCalculator
            task_executor = InferenceCalculator(model_name, partition, compressionProps, use_compression)

        Worker.__init__(self, task_executor=task_executor)

if __name__ == "__main__":
    args = parse_arguments()

    compressionProps = {}
    compressionProps['feature_compression_factor'] = args.feature_compression_factor
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor

    depruneProps = get_depruneProps(args)
    partition = "server" if args.suite in ["fog_offloading", "edge_split"] else "unsplit"
    use_compression = bool(args.use_compression)

    InferenceWorker(model_name=args.model_name,
                    partition=partition,
                    compressionProps=compressionProps,
                    use_compression=use_compression).start()
