#import random
#random.seed(57)
import time

import torch
from torch import nn

from models.neural_network_VGG import NeuralNetwork
from datasets.cifar10 import load_CIFAR10_dataset

from torchvision import models



def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    print(f"Using {device} device")
    return device

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

device = get_device()

'''
model =  models.vgg16(pretrained = False)
input_lastLayer = model.classifier[6].in_features
model.classifier[6] = nn.Linear(input_lastLayer,10)
model = model.to(device)
''' 

model = NeuralNetwork()
input_lastLayer = model.classifier[6].in_features
model.classifier[6] = nn.Linear(input_lastLayer,10)

model = model.to(device)
train_dataloader, test_dataloader, classes = load_CIFAR10_dataset(batch_size = 16)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

# Training
epochs = 50
start_time = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    #test(test_dataloader, model, loss_fn)
print("Done!")
end_time = time.time() 
print("time taken in seconds: ", end_time-start_time)

# torch.save(model.state_dict(), model_path)
# print("Saved PyTorch Model State to {:s}".format(model_path))

'''
# Evaluation
model.eval()
x, y = next(iter(test_dataloader))
with torch.no_grad():
    pred = model(x[0])
    predicted, actual = classes[pred[0].argmax(0)], classes[y[0]]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
'''