from sparse.node.worker import Worker
from sparse.dl.gradient_calculator import GradientCalculator

class SplitTrainingFinal(Worker):
    def __init__(self, model, loss_fn, optimizer, benchmark_log_file_prefix = 'benchmark_split_training'):
        task_executor = GradientCalculator(model=model,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        Worker.__init__(self, task_executor = task_executor, benchmark_log_file_prefix = benchmark_log_file_prefix)

