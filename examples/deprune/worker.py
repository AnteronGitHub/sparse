from sparse_framework.node.worker import Worker

from benchmark import parse_arguments, get_depruneProps

class DepruneWorker(Worker):
    def __init__(self, application, model_name, partition, compressionProps):
        if application == 'learning':
            from gradient_calculator_pruning import GradientCalculatorPruneStep
            task_executor = GradientCalculatorPruneStep(model_name, partition, compressionProps, use_compression=True)
        else:
            from inference_calculator_pruning import InferenceCalculatorPruning
            task_executor = InferenceCalculatorPruning(model_name, partition, compressionProps, use_compression=True)

        Worker.__init__(self, task_executor=task_executor)

if __name__ == "__main__":
    args = parse_arguments()

    compressionProps = {}
    compressionProps['feature_compression_factor'] = args.feature_compression_factor
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor

    depruneProps = get_depruneProps(args)
    partition = "server" if args.suite in ["fog_offloading", "edge_split"] else "unsplit"

    DepruneWorker(application=args.application,
                  model_name=args.model_name,
                  partition=partition,
                  compressionProps=compressionProps).start()
