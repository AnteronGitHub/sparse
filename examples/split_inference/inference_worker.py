from sparse_framework.node.worker import Worker
from sparse_framework.dl.inference_calculator import InferenceCalculator

from benchmark import parse_arguments

class SplitInferenceFinal(Worker):
    def __init__(self, model):
        Worker.__init__(self,
                        task_executor = InferenceCalculator(model))

if __name__ == "__main__":
    args = parse_arguments()
    from datasets import DatasetRepository
    from models import ModelTrainingRepository

    compressionProps = {}
    compressionProps['feature_compression_factor'] = args.feature_compression_factor ### resolution compression factor, compress by how many times
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor ###layer compression factor, reduce by how many times TBD
    partition = "server" if args.suite in ["fog_offloading", "edge_split"] else "unsplit"
    model, loss_fn, optimizer = ModelTrainingRepository().get_model(args.model, partition, compressionProps)

    SplitInferenceFinal(model).start()
