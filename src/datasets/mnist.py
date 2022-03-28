from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def load_mnist_dataset(batch_size = 64):
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]


    return DataLoader(training_data, batch_size=batch_size), DataLoader(test_data, batch_size=batch_size), classes
