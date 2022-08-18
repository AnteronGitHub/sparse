import argparse

import torch
from torch import nn
from tqdm import tqdm

from sparse.stats.training_monitor import TrainingMonitor
from sparse.logging.file_logger import FileLogger

from datasets.cifar10 import load_CIFAR10_dataset
from models.vgg import VGG_unsplit


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', default=64, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--log-file-prefix', default='training-benchmark', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    model = VGG_unsplit()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = args.epochs
    batches = args.batches
    batch_size = args.batch_size
    dataloader, classes = load_CIFAR10_dataset(batch_size)
    print(f"Using {device} for processing")
    print(f"Training unsplit {type(model).__name__} model in {epochs} epochs with {batches*batch_size} samples using batch size {batch_size}")

    progress_bar = tqdm(total=epochs*batches*batch_size, unit='samples', unit_scale=True)
    model.to(device)
    monitor = TrainingMonitor()
    logger = FileLogger(file_prefix=f"{args.log_file_prefix}-{type(model).__name__}-{epochs}-{batch_size}")
    logger.log_row(monitor.get_metrics())
    monitor.start()
    for t in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)

            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.log_row(','.join([str(v) for v in monitor.read_stats(len(X))]))
            progress_bar.update(len(X))
            if batch + 1 >= batches:
                break

    progress_bar.close()
