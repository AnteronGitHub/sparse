
def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='aio', type=str)
    parser.add_argument('--model-name', default='VGG_unsplit', type=str)
    parser.add_argument('--batches', default=64, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--benchmark-node-name', default='client', type=str)
    parser.add_argument('--dataset', default='CIFAR10', type=str, help="Options: FMNIST, CIFAR10, CIFAR100, Imagenet100")
    parser.add_argument('--feature_compression_factor', default=1, type=int)
    parser.add_argument('--resolution_compression_factor', default=1, type=int)

    return parser.parse_args()

def get_depruneProps():
    # depruneProps format is {step, budget, ephochs, pruneState} with all others int and pruneState boolean
    depruneProps = {
                    1: {'budget':16, 'epochs':2, 'pruneState':True},
                    2: {'budget':128, 'epochs':2, 'pruneState':True}
    }
    return depruneProps

def get_deprune_epochs(depruneProps):
    total_epochs = 0
    for entry in depruneProps:
        total_epochs += depruneProps[entry]['epochs']
    return total_epochs

def _get_benchmark_log_file_prefix(args, node_name, epochs):
    return f"learning-{args.suite}-{node_name}-{args.model_name}-{args.dataset}-{epochs}_{args.batch_size}_{args.batch_size}"

def run_aio_benchmark(args):
    print('All-in-one benchmark suite')
    print('--------------------------')

    import asyncio
    from datasets import DatasetRepository
    from models import ModelTrainingRepository
    from nodes.all_in_one import AllInOne

    dataset, classes = DatasetRepository().get_dataset(args.model_name, args.dataset)
    model, loss_fn, optimizer = ModelTrainingRepository().get_model(args.model_name)

    asyncio.run(AllInOne(dataset, classes, model, loss_fn, optimizer).train(args.batches,
                                                                            args.batch_size,
                                                                            args.epochs,
                                                                            _get_benchmark_log_file_prefix(args)))

def run_monitor(args):
    from sparse_framework.stats.monitor_server import MonitorServer
    MonitorServer().start()

if __name__ == '__main__':
    args = parse_arguments()

    run_aio_benchmark(args)
