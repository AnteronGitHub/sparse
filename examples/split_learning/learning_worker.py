from sparse_framework.node.worker import Worker
from sparse_framework.dl.gradient_calculator import GradientCalculator, GradientCalculatorPruneStep

from benchmark import parse_arguments, get_depruneProps, get_deprune_epochs, _get_benchmark_log_file_prefix
from models import ModelTrainingRepository

class LearningWorker(Worker):
    def __init__(self, model, loss_fn, optimizer, depruneProps, benchmark_log_file_prefix):
        task_executor = GradientCalculatorPruneStep(model=model,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           depruneProps = depruneProps)
        Worker.__init__(self, task_executor = task_executor, benchmark_log_file_prefix = benchmark_log_file_prefix)

if __name__ == '__main__':
    args = parse_arguments()

    compressionProps = {}
    ### resolution compression factor, compress by how many times
    compressionProps['feature_compression_factor'] = args.feature_compression_factor
    ###layer compression factor, reduce by how many times TBD
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor

    depruneProps = get_depruneProps()
    partition = "server" if args.suite in ["fog_offloading", "edge_split"] else "unsplit"
    model, loss_fn, optimizer = ModelTrainingRepository().get_model(args.model_name, partition, compressionProps)

    LearningWorker(model=model,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   depruneProps = depruneProps,
                   benchmark_log_file_prefix = _get_benchmark_log_file_prefix(args, "worker", get_deprune_epochs(depruneProps))).start()
