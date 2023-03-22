
class DatasetRepository:
    def get_dataset(self, model_name):
        if model_name == 'VGG_unsplit':
            from .cifar10 import CIFAR10_dataset
            return CIFAR10_dataset()
        elif model_name == 'VGG_client':
            from .cifar10 import CIFAR10_dataset
            return CIFAR10_dataset()
        elif model_name == 'VGG_server':
            from .cifar10 import CIFAR10_dataset
            return CIFAR10_dataset()
        elif model_name == 'Small_unsplit':
            from .mnist_fashion import FashionMNIST
            return FashionMNIST()
        elif model_name == 'Small_client':
            from .mnist_fashion import FashionMNIST
            return FashionMNIST()
        elif model_name == 'Small_server':
            from .mnist_fashion import FashionMNIST
            return FashionMNIST()
        elif model_name == 'YOLOv3' or  model_name == 'YOLOv3_server':
            from .yolov3 import YOLOv3Dataset
            return YOLOv3Dataset()
        else:
            raise f'No model with the name {model_name} was found in the repository'
