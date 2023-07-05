# Sparse

This repository contains source code for Stream Processing Architecture for Resource Subtle Environments (or just
Sparse for short). Additionally, sample applications utilizing Sparse for deep learning can be found in examples
directory.

## Quick start with deep learning

Follow these instructions to start creating your own sparse applications for distributed deep learning with PyTorch.

First, install sparse framework from PyPi:

```
pip install sparse-framework
```

Create a sparse worker node which trains a neural network using data sent by master:
```model_trainer.py
"""model_trainer.py
"""
import torch
from torch import nn

from sparse_framework.node.worker import Worker
from sparse_framework.dl.gradient_calculator import GradientCalculator

# PyTorch model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Sparse node
class ModelTrainer(Worker):
    def __init__(self):
        model = NeuralNetwork()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        Worker.__init__(self,
                        task_executor = GradientCalculator(model=model,
                                                           loss_fn=loss_fn,
                                                           optimizer=optimizer))

if __name__ == '__main__':
    ModelTrainer().start()
```

Then create the corresponding sparse master node:
```data_source.py
"""data_source.py
"""
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import asyncio

from sparse_framework.dl.serialization import encode_offload_request, decode_offload_response
from sparse_framework.node.master import Master

# Sparse node
class TrainingDataSource(Master):
    async def train(self, batch_size = 64, epochs = 1):
        # torchvision dataset
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        for t in range(epochs):
            for batch, (X, y) in enumerate(DataLoader(training_data, batch_size)):
                input_data = encode_offload_request(X, y)
                result_data = await self.task_deployer.deploy_task(input_data)
                split_grad, loss = decode_offload_response(result_data)
                print('Loss: {}'.format(loss))

if __name__ == '__main__':
    asyncio.run(TrainingDataSource().train())
```

To run training, start the worker and the master processes (in separate terminal sessions):
```
python model_trainer.py
python data_source.py
```

## Example Applications

The repository includes example applications (in the [examples directory](./examples)). The applications are tested
tested with the following devices and the following software:

| Device            | JetPack version | Python version | PyTorch version | Docker version | Base image                                     | Docker tag suffix |
| ----------------- | --------------- | -------------- | --------------- | -------------- | ---------------------------------------------- | ------------------ |
| Jetson AGX Xavier | 5.0 preview     | 3.8.10         | 1.12.0a0        | 20.10.12       | nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3 | jp50               |
| Lenovo ThinkPad   | -               | 3.8.12         | 1.11.0          | 20.10.15       | pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime | amd64              |

See [how to deploy the example applications with Kubernetes](./k8s).

