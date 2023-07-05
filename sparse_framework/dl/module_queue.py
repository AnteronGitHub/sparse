from torch.nn import Module, ModuleList

class ModuleQueue(Module):
    def __init__(self, partitions = []):
        super().__init__()
        self.partitions = ModuleList(partitions)

    def forward(self, x):
        for p in self.partitions:
            x = p(x)
        return x

    def append(self, module):
        self.partitions.append(module)

