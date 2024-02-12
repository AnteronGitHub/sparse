import torch

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='aio', type=str)
    parser.add_argument('--model-name', default='VGG_unsplit', type=str)
    parser.add_argument('--dataset', default='CIFAR10', type=str, help="Options: FMNIST, CIFAR10, CIFAR100, Imagenet100")
    parser.add_argument('--no-samples', default=64, type=int)
    parser.add_argument('--target-latency', default=200, type=int)
    parser.add_argument('--use-scheduling', default=1, type=int)
    parser.add_argument('--use-batching', default=1, type=int)
    parser.add_argument('--application', default='learning', type=str)
    parser.add_argument('--no-datasources', default=1, type=int)
    parser.add_argument('--no-models', default=1, type=int)

    return parser.parse_args()

def _get_benchmark_log_file_prefix(args, node_name):
    return f"splitnn-{args.application}-{args.suite}-{node_name}-{args.model_name}-{args.dataset}_{args.no_samples}"

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def count_model_parameters(model):
    num_parameters = 0
    for param in model.parameters():
        num_parameters += param.nelement()
    return num_parameters

