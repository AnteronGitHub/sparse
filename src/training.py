import torch
from torch import nn

from models.neural_network import NeuralNetwork
from datasets.mnist_fashion import load_mnist_fashion_dataset

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing {device} device")
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

def train_model(model, train_dataloader, loss_fn, optimizer, test_dataloader, epochs = 5):
    print("\nTraining model")
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

def evaluate_model(model, dataloader, classes):
    print("\nEvaluating model")
    model.eval()
    x, y = next(iter(dataloader))
    with torch.no_grad():
        pred = model(x[0])
        predicted, actual = classes[pred[0].argmax(0)], classes[y[0]]
        print(f'Predicted: "{predicted}", Actual: "{actual}"\n')

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print("Saved PyTorch Model State to {:s}".format(filepath))

if __name__ == "__main__":
    device = get_device()

    model = NeuralNetwork().to(device)
    train_dataloader, test_dataloader, classes = load_mnist_fashion_dataset()

    train_model(model,
                train_dataloader,
                nn.CrossEntropyLoss(),
                torch.optim.SGD(model.parameters(), lr=1e-3),
                test_dataloader)

    evaluate_model(model, test_dataloader, classes)

