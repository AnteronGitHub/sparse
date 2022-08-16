import torch
from torch import nn
from tqdm import tqdm

from datasets.cifar10 import load_CIFAR10_dataset
from models.vgg import NeuralNetwork

if __name__ == "__main__":
    model = NeuralNetwork()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 1
    dataloader, classes = load_CIFAR10_dataset()
    print(f"Using {device} for processing")
    print(f"Training unsplit {type(model).__name__} model in {epochs} epochs with {len(dataloader.dataset)} samples")
    progress_bar = tqdm(total=epochs*len(dataloader.dataset), unit='samples', unit_scale=True)

    model.to(device)
    for t in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)

            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(len(X))

        progress_bar.close()
