from sparse_framework import Worker

from utils import parse_arguments, _get_benchmark_log_file_prefix

class SplitNNWorker(Worker):
    def __init__(self, application, benchmark_log_file_prefix):
        if application == 'learning':
            from gradient_calculator import GradientCalculator
            task_executor = GradientCalculator(capacity = 0)
        else:
            from inference_calculator import InferenceCalculator
            task_executor = InferenceCalculator()
        Worker.__init__(self, task_executor = task_executor, benchmark_log_file_prefix = benchmark_log_file_prefix)

if __name__ == '__main__':
    args = parse_arguments()

    benchmark_log_file_prefix = _get_benchmark_log_file_prefix(args, "worker")

    SplitNNWorker(application=args.application,
                  benchmark_log_file_prefix=benchmark_log_file_prefix).start()
