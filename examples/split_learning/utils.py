
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
    parser.add_argument('--deprune-props',
                        type=str,
                        default="budget:16;epochs:2;pruneState:1,budget:128;epochs:2;pruneState:1",
                        help="Comma separated list of phases in format: budget:int;epochs:int;pruneState:[01]")

    return parser.parse_args()

def get_depruneProps(args):
    depruneProps = []
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

def get_deprune_epochs(depruneProps):
    total_epochs = 0
    for prop in depruneProps:
        total_epochs += prop['epochs']
    return total_epochs

def _get_benchmark_log_file_prefix(args, node_name, epochs):
    return f"learning-{args.suite}-{node_name}-{args.model_name}-{args.dataset}-{epochs}_{args.batch_size}_{args.batch_size}"
