from sparse_framework import Worker

from batched_rx_pipe import BatchedRXPipe
from utils import parse_arguments, _get_benchmark_log_file_prefix

class SplitNNWorker(Worker):
    def __init__(self, application, benchmark_log_file_prefix, use_batching = True):
        if application == 'learning':
            from gradient_calculator import GradientCalculator
            task_executor = GradientCalculator(capacity = 0)
        else:
            from inference_calculator import InferenceCalculator
            task_executor = InferenceCalculator()
        if use_batching:
            rx_pipe = BatchedRXPipe(benchmark_log_file_prefix = benchmark_log_file_prefix)
        else:
            rx_pipe = None

        Worker.__init__(self,
                        task_executor,
                        rx_pipe,
                        benchmark_log_file_prefix)

        if use_batching:
            self.logger.info("Batching requests in rx pipe")
        else:
            self.logger.info("Not batching requests in rx pipe")

if __name__ == '__main__':
    args = parse_arguments()

    benchmark_log_file_prefix = _get_benchmark_log_file_prefix(args, "worker")
    use_batching = args.batch_size > 1

    SplitNNWorker(application=args.application,
                  benchmark_log_file_prefix=benchmark_log_file_prefix,
                  use_batching=use_batching).start()
