from sparse_framework.node.master import Master
from sparse_framework.node.worker import Worker

from utils import parse_arguments, _get_benchmark_log_file_prefix

class SplitNNIntermediate(Master, Worker):
    def __init__(self, application, model_name, partition, benchmark_log_file_prefix):
        if application == 'learning':
            from gradient_calculator import GradientCalculator
            task_executor = GradientCalculator(model_name, partition, compressionProps={}, use_compression=False)
        else:
            from inference_calculator import InferenceCalculator
            task_executor = InferenceCalculator(model_name, partition, compressionProps={}, use_compression=False)
        Master.__init__(self)
        Worker.__init__(self, task_executor = task_executor, benchmark_log_file_prefix = benchmark_log_file_prefix)

if __name__ == '__main__':
    args = parse_arguments()

    SplitNNIntermediate(application=args.application,
                        model_name=args.model_name,
                        partition="client",
                        benchmark_log_file_prefix=_get_benchmark_log_file_prefix(args, "intermediate")).start()
