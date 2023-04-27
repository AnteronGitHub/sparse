import argparse
import asyncio

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='aio', type=str)
    parser.add_argument('--benchmark-node-name', default='amd64', type=str)
    parser.add_argument('--model_name', default='VGG_unsplit', type=str)   #YOLOv3_server prev, this needs to be worked on
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
-{args.model_name}-{args.dataset}-{args.batch_size}-{args.resolution_compression_factor}"

def run_aio_benchmark(args):
    print('All-in-one benchmark suite')
    print('--------------------------')

    from nodes.all_in_one import AllInOne

    AllInOne().benchmark(inferences_to_be_run = args.inferences_to_be_run)

def run_offload_client_benchmark(args):
    print('Offload client node benchmark suite')
    print('-----------------------------------')

    from nodes.split_inference_client import SplitInferenceClient
    from datasets import DatasetRepository
    from models import ModelTrainingRepository

    dataset, classes = DatasetRepository().get_dataset(args.model_name, args.dataset)
    compressionProps = {} # resolution compression factor, compress by how many times
    compressionProps['feature_compression_factor'] = args.feature_compression_factor # layer compression factor, reduce by how many times TBD
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor
    model, loss_fn, optimizer = ModelTrainingRepository().get_model(args.model_name, compressionProps)
    depruneProps = get_depruneProps()
    asyncio.run(SplitInferenceClient(dataset, model).infer(args.batch_size,
                                                                        args.batches,
                                                                            depruneProps,
                                                           log_file_prefix=_get_benchmark_log_file_prefix(args)))

def run_offload_datasource_benchmark(args):
    print('Offload data source benchmark suite')
    print('-----------------------------------')

    import asyncio
    from datasets import DatasetRepository
    from nodes.inference_data_source import InferenceDataSource
    from models import ModelTrainingRepository
    
    dataset, classes = DatasetRepository().get_dataset(args.model_name, args.dataset)
    depruneProps = get_depruneProps()
    asyncio.run(InferenceDataSource(dataset, args.model_name).start(args.batch_size,
                                                                        args.batches,
                                                                            depruneProps,
                                                                    log_file_prefix=_get_benchmark_log_file_prefix(args)))

def run_offload_intermediate_benchmark(args):
    print('Offload intermediate node benchmark suite')
    print('-----------------------------------------')

    from nodes.split_inference_intermediate import SplitInferenceIntermediate
    from datasets import DatasetRepository
    from models import ModelTrainingRepository
    
    compressionProps = {}
    compressionProps['feature_compression_factor'] = args.feature_compression_factor ### resolution compression factor, compress by how many times
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor ###layer compression factor, reduce by how many times TBD
    model, loss_fn, optimizer = ModelTrainingRepository().get_model(args.model_name, compressionProps)
    
    SplitInferenceIntermediate(model).start()

def run_offload_final_benchmark(args):
    print('Offload final node benchmark suite')
    print('----------------------------------')

    from nodes.split_inference_final import SplitInferenceFinal
    from datasets import DatasetRepository
    from models import ModelTrainingRepository

    compressionProps = {}
    compressionProps['feature_compression_factor'] = args.feature_compression_factor ### resolution compression factor, compress by how many times
    compressionProps['resolution_compression_factor'] = args.resolution_compression_factor ###layer compression factor, reduce by how many times TBD
    model, loss_fn, optimizer = ModelTrainingRepository().get_model(args.model_name, compressionProps)

    if args.model_name == "YOLOv3_server":
        from models.yolov3 import YOLOv3_server
        model = YOLOv3_server(compressionProps = compressionProps)
    elif args.model_name == "YOLOv3":
        from models.yolov3 import YOLOv3
        model = YOLOv3(compressionProps = compressionProps)
        
    SplitInferenceFinal(model).start()

def run_monitor(args):
    from sparse_framework.stats.monitor_server import MonitorServer
    MonitorServer().start()

if __name__ == "__main__":
    args = parse_arguments()
    if args.suite == 'aio':
        run_aio_benchmark(args)
    elif args.suite == 'offload_source':
        run_offload_datasource_benchmark(args)
    elif args.suite == 'offload_client':
        run_offload_client_benchmark(args)
    elif args.suite == 'offload_intermediate':
        run_offload_intermediate_benchmark(args)
    elif args.suite == 'offload_final':
        run_offload_final_benchmark(args)
    elif args.suite == 'monitor':
        run_monitor(args)
    else:
        print(f"Available suites: 'aio', 'offload_final', 'offload_client', 'monitor'")
