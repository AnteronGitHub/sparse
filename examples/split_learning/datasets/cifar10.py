from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def load_CIFAR10_dataset(batch_size = 64):
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform= transforms.Compose([
                    transforms.Resize(size=(32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize( 
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
                    )
                ])   
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose([
                    transforms.Resize(size=(32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize( 
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
                    )
                ])   
    )

    classes = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck']


    return DataLoader(training_data, batch_size=batch_size), DataLoader(test_data, batch_size=batch_size), classes
