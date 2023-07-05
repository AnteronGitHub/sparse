from torchvision import datasets
from torchvision.transforms import ToTensor

def FashionMNIST():
    dataset = datasets.FashionMNIST(
        root="/data",
        train=True,
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


    return dataset, classes
