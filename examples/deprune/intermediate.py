from sparse_framework import Master, Worker

from benchmark import parse_arguments, get_depruneProps, _get_benchmark_log_file_prefix

class DepruneIntermediate(Master, Worker):
    def __init__(self, application, model_name, partition, compressionProps, benchmark_log_file_prefix):
        if application == 'learning':
            from gradient_calculator_pruning import GradientCalculatorPruneStep
            task_executor = GradientCalculatorPruneStep(model_name, partition, compressionProps)
        else:
            from inference_calculator_pruning import InferenceCalculatorPruning
            task_executor = InferenceCalculatorPruning(model_name, partition, compressionProps)
        Master.__init__(self)
        Worker.__init__(self, task_executor = task_executor, benchmark_log_file_prefix = benchmark_log_file_prefix)

if __name__ == "__main__":
    args = parse_arguments()

    compressionProps = {}
    compressionProps['feature_compression_factor'] = args.feature_compression_factor
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor

    depruneProps = get_depruneProps(args)
    partition = "client"

    DepruneIntermediate(application=args.application,
                        model_name=args.model_name,
                        partition=partition,
                        compressionProps=compressionProps,
                        benchmark_log_file_prefix=_get_benchmark_log_file_prefix(args, "intermediate")).start()
