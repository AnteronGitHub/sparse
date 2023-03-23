from sparse_framework.node.worker import Worker
from sparse_framework.dl.gradient_calculator import GradientCalculator, GradientCalculatorPruneStep

class SplitTrainingFinal(Worker):
    def __init__(self, model, loss_fn, optimizer, depruneProps, benchmark_log_file_prefix = 'benchmark_split_training'):
        task_executor = GradientCalculatorPruneStep(model=model,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           depruneProps = depruneProps)
        Worker.__init__(self, task_executor = task_executor, benchmark_log_file_prefix = benchmark_log_file_prefix)

