import torch
from torch import nn

from models.neural_network2 import NeuralNetwork_local
from datasets.mnist_fashion import load_mnist_fashion_dataset

from networking import run_offload_training
from utils import get_device

def train_epoch(dataloader, model_local, loss_fn, optimizer_local):
    size = len(dataloader.dataset)
    model_local.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to('cpu')

        # Local forward propagation
        split_vals = model_local(X)

        # Offloaded layers
        np_split_grad, loss = run_offload_training(split_vals.detach().numpy(), y.detach().numpy())

        # Local back propagation
        split_grad = torch.from_numpy(np_split_grad).to('cpu')
        optimizer_local.zero_grad()
        split_vals.backward(split_grad)
        optimizer_local.step()

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# TODO: Implement evaluation over network
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

# TODO: Implement evaluation over network
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

# torch.save(model.state_dict(), model_path)
# print("Saved PyTorch Model State to {:s}".format(model_path))

# evaluate_model(model1, model2, test_dataloader)
