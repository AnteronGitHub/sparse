from time import time

import torch

from sparse_framework import SparseIOBuffer

from utils import get_device

class MemoryBuffer(SparseIOBuffer):
    """Memory buffer for Node's task executor.

    Buffer ensures that when a task is created for the executor, all of the needed data are available in the executor
    device memory. After the executor task has been processed, memory buffer transfers the data to the network device
    for result transmission.
    """
    def __init__(self):
        super().__init__()

        self.device = get_device()

    def transferToDevice(self, tensor):
        return tensor.to(self.device)

    def transferToHost(self, tensor):
        return tensor.to("cpu")

