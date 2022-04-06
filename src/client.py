import torch
from torch import nn

from datasets.mnist_fashion import load_mnist_fashion_dataset
from models.neural_network import NeuralNetwork_local
from training import train_epoch
from utils import get_device

if __name__ == '__main__':
    device = get_device()
    model1 = NeuralNetwork_local().to(device)
    train_dataloader, test_dataloader, classes = load_mnist_fashion_dataset()

    loss_fn = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=1e-3)

    # Training
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(train_dataloader, model1, loss_fn, optimizer1)
        # test(test_dataloader, model1, model2, loss_fn)
    print("Done!")

    # evaluate_model(model1, model2, test_dataloader)
