import torch
from torch import nn

from datasets.mnist_fashion import load_mnist_fashion_dataset
from models.neural_network import NeuralNetwork_local
from roles.master import Master
from training import train_epoch
from utils import get_device

class SplitTrainingClient(Master):
    def __init__(self):
        self.device = get_device()
        self.model = NeuralNetwork_local().to(self.device)
        self.train_dataloader, self.test_dataloader, self.classes = load_mnist_fashion_dataset()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    def train(self, epochs : int = 5):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_epoch(self.train_dataloader, self.model, self.loss_fn, self.optimizer)

            # test(test_dataloader, model1, model2, loss_fn)
        print("Done!")

        # evaluate_model(model1, model2, test_dataloader)

if __name__ == '__main__':
    SplitTrainingClient().train()
