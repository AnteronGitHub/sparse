import argparse
import asyncio

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='aio', type=str)
    parser.add_argument('--benchmark-node-name', default='amd64', type=str)
    parser.add_argument('--model', default='YOLOv3_server', type=str)
    parser.add_argument('--inferences-to-be-run', default=100, type=int)
    
    parser.add_argument('--feature_compression_factor', default=1, type=int)
    parser.add_argument('--resolution_compression_factor', default=1, type=int)
    
    return parser.parse_args()

def run_aio_benchmark(args):
    print('All-in-one benchmark suite')
    print('--------------------------')

    from nodes.all_in_one import AllInOne

    AllInOne().benchmark(inferences_to_be_run = args.inferences_to_be_run)

def run_offload_client_benchmark(args):
    print('Offload client node benchmark suite')
    print('-----------------------------------')

    from nodes.split_inference_client import SplitInferenceClient

    asyncio.run(SplitInferenceClient().infer())

def run_offload_datasource_benchmark(args):
    print('Offload data source benchmark suite')
    print('-----------------------------------')

    import asyncio
    from nodes.inference_data_source import InferenceDataSource

    asyncio.run(InferenceDataSource().start())

def run_offload_intermediate_benchmark(args):
    print('Offload intermediate node benchmark suite')
    print('-----------------------------------------')

    from nodes.split_inference_intermediate import SplitInferenceIntermediate
    SplitInferenceIntermediate().start()

def run_offload_final_benchmark(args):
    print('Offload final node benchmark suite')
    print('----------------------------------')

    from nodes.split_inference_final import SplitInferenceFinal
    from models import ModelTrainingRepository

    compressionProps = {}
    compressionProps['feature_compression_factor'] = 4
    compressionProps['resolution_compression_factor'] = 1

    if args.model == "YOLOv3_server":
        from models.yolov3 import YOLOv3_server
        model = YOLOv3_server(compressionProps = compressionProps)
    elif args.model == "YOLOv3":
        from models.yolov3 import YOLOv3
        model = YOLOv3(compressionProps = compressionProps)
        
    if args.model == "VGG":
        model, _, _ = ModelTrainingRepository().get_model(args.model_name, compressionProps)    

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
