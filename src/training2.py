import torch
from torch import nn
from torch.autograd import Variable

from models.neural_network2 import NeuralNetwork
from models.neural_network2 import NeuralNetwork_local
from models.neural_network2 import NeuralNetwork_server
from datasets.mnist_fashion import load_mnist_fashion_dataset

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device

def offload_layers(model, split_activation, y, loss_fn, optimizer):
    split_activation = Variable(split_activation, requires_grad=True).to('cpu')
    pred = model(split_activation)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    gradient = split_activation.grad.to(device)
    optimizer.step()

    return gradient, loss

def train_epoch(dataloader, model_local, model_server, loss_fn, optimizer_local, optimizer_server):
    size = len(dataloader.dataset)
    model_local.train()
    model_server.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to('cpu')

        # Local forward propagation
        split_vals = model_local(X)

        # Offloaded layers
        split_grad, loss = offload_layers(model_server,
                                          split_vals,
                                          y,
                                          loss_fn,
                                          optimizer_server)

        # Local back propagation
        optimizer_local.zero_grad()
        split_vals.backward(split_grad)
        optimizer_local.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model1, model2, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model1.eval()
    model2.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model2(model1(X))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def evaluate_model(model1, model2, dataloader):
    model1.eval()
    model2.eval()
    x, y = next(iter(test_dataloader))
    with torch.no_grad():
        pred = model2(model1(x[0]))
        predicted, actual = classes[pred[0].argmax(0)], classes[y[0]]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

device = get_device()
model1 = NeuralNetwork_local().to(device)
model2 = NeuralNetwork_server().to('cpu')
train_dataloader, test_dataloader, classes = load_mnist_fashion_dataset()

loss_fn = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=1e-3)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-3)

# Training
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_epoch(train_dataloader, model1, model2, loss_fn, optimizer1, optimizer2)
    test(test_dataloader, model1, model2, loss_fn)
print("Done!")

# torch.save(model.state_dict(), model_path)
# print("Saved PyTorch Model State to {:s}".format(model_path))

evaluate_model(model1, model2, test_dataloader)
