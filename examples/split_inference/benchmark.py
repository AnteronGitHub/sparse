import argparse
import asyncio

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='aio', type=str)
    parser.add_argument('--benchmark-node-name', default='amd64', type=str)
    parser.add_argument('--model', default='VGG_unsplit', type=str)   #YOLOv3_server prev, this needs to be worked on
    parser.add_argument('--inferences-to-be-run', default=100, type=int)
    parser.add_argument('--batches', default=64, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--dataset', default='CIFAR10', type=str, help="Options: FMNIST, CIFAR10, CIFAR100, Imagenet100")
    parser.add_argument('--feature_compression_factor', default=1, type=int)
    parser.add_argument('--resolution_compression_factor', default=1, type=int)
    
    return parser.parse_args()

def get_depruneProps(): 
    # depruneProps format is {step, budget, ephochs, pruneState} with all others int and pruneState boolean
    depruneProps = {
                'budget':16, 'epochs':2, 'pruneState':True, 
    }
    return depruneProps

def _get_benchmark_log_file_prefix(args):
    return f"benchmark_inference-{args.suite}-{args.benchmark_node_name}\
-{args.model}-{args.dataset}-{args.batch_size}-{args.resolution_compression_factor}"

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
