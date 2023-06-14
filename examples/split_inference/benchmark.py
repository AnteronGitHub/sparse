import argparse
import asyncio

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='aio', type=str)
    parser.add_argument('--model-name', default='VGG', type=str)   #YOLOv3_server prev, this needs to be worked on
    parser.add_argument('--inferences-to-be-run', default=100, type=int)
    parser.add_argument('--batches', default=64, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--dataset', default='CIFAR10', type=str, help="Options: FMNIST, CIFAR10, CIFAR100, Imagenet100")
    parser.add_argument('--feature_compression_factor', default=1, type=int)
    parser.add_argument('--resolution_compression_factor', default=1, type=int)
    parser.add_argument('--use-compression', default=1, type=int)
    parser.add_argument('--deprune-props',
                        type=str,
                        default="budget:16;epochs:2;pruneState:1,budget:128;epochs:2;pruneState:1",
                        help="Comma separated list of phases in format: budget:int;epochs:int;pruneState:[01]")

    return parser.parse_args()

def get_depruneProps(args):
    depruneProps = []
    if len(args.deprune_props) == 0:
        return depruneProps
    for phase in args.deprune_props.split(","):
        prop = {}
        for phase_prop in phase.split(";"):
            [k, v] = phase_prop.split(":")
            if k in ["budget", "epochs"]:
                prop[k] = int(v)
            else:
                prop[k] = bool(int(v))
        depruneProps.append(prop)
    return depruneProps

def _get_benchmark_log_file_prefix(args):
    return f"benchmark_inference-{args.suite}\
-{args.model_name}-{args.dataset}-{args.batch_size}-{args.resolution_compression_factor}"

def run_offload_intermediate_benchmark(args):
    print('Offload intermediate node benchmark suite')
    print('-----------------------------------------')

    from nodes.split_inference_intermediate import SplitInferenceIntermediate
    from datasets import DatasetRepository
    from models import ModelTrainingRepository
    
    compressionProps = {}
    compressionProps['feature_compression_factor'] = args.feature_compression_factor ### resolution compression factor, compress by how many times
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor ###layer compression factor, reduce by how many times TBD
    model, loss_fn, optimizer = ModelTrainingRepository().get_model(args.model, compressionProps)
    
    SplitInferenceIntermediate(model).start()

def run_monitor(args):
    from sparse_framework.stats.monitor_server import MonitorServer
    MonitorServer().start()

if __name__ == "__main__":
    args = parse_arguments()
    if args.suite == 'offload_intermediate':
        run_offload_intermediate_benchmark(args)
    elif args.suite == 'monitor':
        run_monitor(args)
    else:
        print(f"Available suites: 'aio', 'offload_final', 'offload_client', 'monitor'")
