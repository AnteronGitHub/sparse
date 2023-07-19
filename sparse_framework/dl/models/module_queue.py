import torch
from torch.nn import Module, ModuleList

import os

class ModuleQueue(Module):
    def __init__(self, partitions = [], start = 0):
        super().__init__()
        self.partitions = ModuleList(partitions)
        self.start = start

    def forward(self, x):
        for p in self.partitions:
            x = p(x)
        return x

    def append(self, module):
        self.partitions.append(module)

    def pop(self):
        """Removes the first module in the queue, increasing the start index by one.
        """
        module = self.partitions.__getitem__(0)
        self.partitions.__delitem__(0)
        self.start += 1
        return module

    def load_parameters(self, data_path : str, model_name : str):
        for i, p in enumerate(self.partitions):
            filepath = os.path.join(data_path, f"{model_name}_{self.start+i}.pt")
            if not os.path.exists(filepath):
                return False
            p.load_state_dict(torch.load(filepath))

        return True

    def save_parameters(self, data_path : str, model_name : str):
        for i, p in enumerate(self.partitions):
            filepath = os.path.join(data_path, f"{model_name}_{self.start+i}.pt")
            torch.save(p.state_dict(), filepath)
