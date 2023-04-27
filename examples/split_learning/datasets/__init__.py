
class DatasetRepository:
    def get_dataset(self, model_name, dataset):
        if model_name in ['VGG_unsplit', 'VGG_client', 'VGG_server']:
            if dataset == 'CIFAR10':
                from .cifar10 import CIFAR10_dataset
                return CIFAR10_dataset()
            elif dataset == 'CIFAR100':
                from .cifar100 import CIFAR100_dataset
                return CIFAR100_dataset()
            elif dataset == 'Imagenet100':
                from .imagenet100 import Imagenet100_dataset
                return Imagenet100_dataset()
            else:
                raise f'No dataset with the name {dataset} was found in the repository'
        elif model_name in ['Small_unsplit', 'Small_client', 'Small_server']:
            from .mnist_fashion import FashionMNIST
            return FashionMNIST()
        else:
            raise f'No model with the name {model_name} was found in the repository'
