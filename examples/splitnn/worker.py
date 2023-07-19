from sparse_framework import Worker
from sparse_framework.dl import ModelTrainingRepository

from utils import parse_arguments, _get_benchmark_log_file_prefix

class SplitNNWorker(Worker):
    def __init__(self, application, model_name, partition, benchmark_log_file_prefix):
        if application == 'learning':
            from gradient_calculator import GradientCalculator
            task_executor = GradientCalculator(capacity = 0, model_name=model_name, partition=partition)
        else:
            from inference_calculator import InferenceCalculator
            task_executor = InferenceCalculator(model_name, partition)
        Worker.__init__(self, task_executor = task_executor, benchmark_log_file_prefix = benchmark_log_file_prefix)

if __name__ == '__main__':
    args = parse_arguments()

    partition = "server" if args.suite in ["fog_offloading", "edge_split"] else "unsplit"
    benchmark_log_file_prefix = _get_benchmark_log_file_prefix(args, "worker")

    SplitNNWorker(application=args.application,
                  model_name=args.model_name,
                  partition=partition,
                  benchmark_log_file_prefix=benchmark_log_file_prefix).start()
