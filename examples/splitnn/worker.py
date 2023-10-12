import asyncio
import pickle

from sparse_framework import Worker, TCPServer, ConfigManager
from sparse_framework.dl import InMemoryModelRepository, ModelPipe, get_device

from batched_rx_pipe import BatchedRXPipe
from utils import parse_arguments, _get_benchmark_log_file_prefix

class SplitNNWorker(Worker):
    def __init__(self, application, benchmark_log_file_prefix, use_batching = True):
        if application == 'learning':
            from gradient_calculator import GradientCalculator
            task_executor = GradientCalculator(get_device(), capacity = 0)
        else:
            from inference_calculator import InferenceCalculator
            task_executor = InferenceCalculator()
        self.model_repository = None
        Worker.__init__(self, task_executor, lambda: ModelPipe(task_executor, self.model_repository))

    def start(self):
        self.model_repository = InMemoryModelRepository(self, get_device())
        super().start()

if __name__ == '__main__':
    args = parse_arguments()

    benchmark_log_file_prefix = _get_benchmark_log_file_prefix(args, "worker")
    use_batching = args.batch_size > 1

    SplitNNWorker(application=args.application,
                  benchmark_log_file_prefix=benchmark_log_file_prefix,
                  use_batching=use_batching).start()
