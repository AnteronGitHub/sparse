import torch
from torch import nn

from networking import run_offload_training

def train_epoch(dataloader, model_local, loss_fn, optimizer_local, device = 'cpu'):
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
def test(dataloader, model1, model2, loss_fn, device = 'cpu'):
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

