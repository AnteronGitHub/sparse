from sparse_framework.node.master import Master
from sparse_framework.node.worker import Worker

from utils import parse_arguments, get_depruneProps, get_deprune_epochs, _get_benchmark_log_file_prefix

class LearningIntermediate(Master, Worker):
    def __init__(self, model_name, partition, compressionProps, benchmark_log_file_prefix, use_compression):
        if use_compression:
            from sparse_framework.dl import GradientCalculatorPruneStep
            task_executor = GradientCalculatorPruneStep(model_name, partition, compressionProps, use_compression)
        else:
            from sparse_framework.dl import GradientCalculator
            task_executor = GradientCalculator(model_name, partition, compressionProps, use_compression)
        Master.__init__(self)
        Worker.__init__(self, task_executor = task_executor, benchmark_log_file_prefix = benchmark_log_file_prefix)

if __name__ == '__main__':
    args = parse_arguments()

    compressionProps = {}
    compressionProps['feature_compression_factor'] = args.feature_compression_factor
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor

    depruneProps = get_depruneProps(args)
    partition = "client"
    use_compression = bool(args.use_compression)
    benchmark_log_file_prefix = _get_benchmark_log_file_prefix(args, "intermediate", get_deprune_epochs(depruneProps))

    LearningIntermediate(model_name=args.model_name,
                         partition=partition,
                         compressionProps=compressionProps,
                         benchmark_log_file_prefix=benchmark_log_file_prefix,
                         use_compression=use_compression).start()
