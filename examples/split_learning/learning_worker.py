from sparse_framework.node.worker import Worker

from utils import parse_arguments, get_depruneProps, get_deprune_epochs, _get_benchmark_log_file_prefix
from models import ModelTrainingRepository

class LearningWorker(Worker):
    def __init__(self, model, loss_fn, optimizer, benchmark_log_file_prefix, use_compression):
        if use_compression:
            from sparse_framework.dl import GradientCalculatorPruneStep
            task_executor = GradientCalculatorPruneStep(model=model,
                                                        loss_fn=loss_fn,
                                                        optimizer=optimizer)
        else:
            from sparse_framework.dl import GradientCalculator
            task_executor = GradientCalculator(model=model,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer)
        Worker.__init__(self, task_executor = task_executor, benchmark_log_file_prefix = benchmark_log_file_prefix)

if __name__ == '__main__':
    args = parse_arguments()

    compressionProps = {}
    ### resolution compression factor, compress by how many times
    compressionProps['feature_compression_factor'] = args.feature_compression_factor
    ###layer compression factor, reduce by how many times TBD
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor

    depruneProps = get_depruneProps(args)
    partition = "server" if args.suite in ["fog_offloading", "edge_split"] else "unsplit"
    model, loss_fn, optimizer = ModelTrainingRepository().get_model(args.model_name, partition, compressionProps)
    benchmark_log_file_prefix = _get_benchmark_log_file_prefix(args, "worker", get_deprune_epochs(depruneProps))

    LearningWorker(model=model,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   benchmark_log_file_prefix=benchmark_log_file_prefix,
                   use_compression=bool(args.use_compression)).start()
