from torchvision import datasets
from torchvision import transforms

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
