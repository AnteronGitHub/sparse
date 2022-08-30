import argparse
import asyncio

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='aio', type=str)
    parser.add_argument('--benchmark-node-name', default='amd64', type=str)
    parser.add_argument('--inferences-to-be-run', default=100, type=int)
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

def run_offload_final_benchmark(args):
    print('Offload final node benchmark suite')
    print('----------------------------------')

    from nodes.split_inference_final import SplitInferenceFinal

    SplitInferenceFinal().start()

def run_monitor(args):
    from sparse.stats.monitor_server import MonitorServer
    MonitorServer().start()

if __name__ == "__main__":
    args = parse_arguments()
    if args.suite == 'aio':
        run_aio_benchmark(args)
    elif args.suite == 'offload_final':
        run_offload_final_benchmark(args)
    elif args.suite == 'offload_client':
        run_offload_client_benchmark(args)
    elif args.suite == 'monitor':
        run_monitor(args)
    else:
        print(f"Available suites: 'aio', 'offload_final', 'offload_client', 'monitor'")
