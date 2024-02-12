from torchvision import datasets
from torchvision import transforms

def FashionMNIST():
    dataset = datasets.FashionMNIST(
        root="/data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
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

def CIFAR10_dataset():
    dataset = datasets.CIFAR10(
            root="/data",
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


    return dataset, classes

def CIFAR100_dataset():
    dataset = datasets.CIFAR100(
            root="/data",
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

    classes = [
        "beaver", "dolphin", "otter", "seal", "whale",
        "aquarium fish", "flatfish", "ray", "shark", "trout",
        "orchids", "poppies", "roses", "sunflowers", "tulips",
        "bottles", "bowls", "cans", "cups", "plates",
        "apples", "mushrooms", "oranges", "pears", "sweet peppers",
        "clock", "computer keyboard", "lamp", "telephone", "television",
        "bed", "chair", "couch", "table", "wardrobe",
        "bee", "beetle", "butterfly", "caterpillar", "cockroach",
        "bear", "leopard", "lion", "tiger", "wolf",
        "bridge", "castle", "house", "road", "skyscraper",
        "cloud", "forest", "mountain", "plain", "sea",
        "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
        "fox", "porcupine", "possum", "raccoon", "skunk",
        "crab", "lobster", "snail", "spider", "worm",
        "baby", "boy", "girl", "man", "woman",
        "crocodile", "dinosaur", "lizard", "snake", "turtle",
        "hamster", "mouse", "rabbit", "shrew", "squirrel",
        "maple", "oak", "palm", "pine", "willow",
        "bicycle", "bus", "motorcycle", "pickup truck", "train",
        "lawn-mower", "rocket", "streetcar", "tank", "tractor"
    ]


    return dataset, classes

# Imagenet100 implementation taken from
# https://github.com/guglielmocamporese/relvit/blob/main/datasets/dataset_option/imagenet100.py

##################################################
# Imports
##################################################

from torch.utils.data import Dataset
import os
import glob
import json
from PIL import Image

from torch.utils.data import DataLoader

##################################################
# Imagenet100 Dataset
##################################################

class Imagenet100(Dataset):
    """
    Subset of the ImageNet dataset.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Imagenet100, self).__init__()
        self.data_dir = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if download:
            self._download()
        self.labels_list = self._retrieve_labels_list()
        self.image_paths, self.labels = self._get_data()

    def _download(self):
        url = 'https://www.kaggle.com/datasets/ambityga/imagenet100/download'
        if not os.path.exists(f'{self.data_dir}/imagenet100/archive.zip'):
            raise Exception(f'Error. Download of Dataset Imagenet100 not yet implemented.' +
                            f'Please download it from {url}')

    def _retrieve_labels_list(self):
        f = open(f'{self.data_dir}/imagenet100/Labels.json', 'r')
        labels_info = json.load(f)
        labels_list = list(labels_info.keys())
        return labels_list

    def _get_data(self):
        image_paths, labels = [], []

        # If train
        if self.train:
            train_sets = glob.glob1(f'{self.data_dir}/imagenet100/', 'train*')
            for train_set in train_sets:
                img_folders = os.listdir(
                    f'{self.data_dir}/imagenet100/{train_set}/')
                for folder in img_folders:
                    list_images = os.listdir(
                        f'{self.data_dir}/imagenet100/{train_set}/{folder}/')
                    label = self.labels_list.index(folder)
                    for img in list_images:
                        image_path = f'{self.data_dir}/imagenet100/{train_set}/{folder}/{img}'
                        image_paths += [image_path]
                        labels += [label]

        # If validation
        else:
            val_folders = os.listdir(f'{self.data_dir}/imagenet100/val.X/')
            for folder in val_folders:
                list_images = os.listdir(
                    f'{self.data_dir}/imagenet100/val.X/{folder}/')
                label = self.labels_list.index(folder)
                for img in list_images:
                    image_path = f'{self.data_dir}/imagenet100/val.X/{folder}/{img}'
                    image_paths += [image_path]
                    labels += [label]

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


##################################################
# Dataloaders
##################################################


def Imagenet100_dataset():
    num_classes = 100

    training_data = Imagenet100(
        root="/data",
        train=True,
        download=False,
        transform=transforms.Compose([
            lambda x: x.convert('RGB'),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4595, 0.4520, 0.3900), (0.2547, 0.2385, 0.2548)
            ),
        ])
    )

    return training_data, num_classes

def get_dataset(dataset_name):
    if dataset_name == 'CIFAR10':
        return CIFAR10_dataset()
    elif dataset_name == 'CIFAR100':
        from .cifar100 import CIFAR100_dataset
        return CIFAR100_dataset()
    elif dataset_name == 'Imagenet100':
        from .imagenet100 import Imagenet100_dataset
        return Imagenet100_dataset()
    elif dataset_name == 'FMNIST':
        from .mnist_fashion import FashionMNIST
        return FashionMNIST()
    else:
        raise f'No dataset with the name {dataset} was found in the repository'

