import torch.nn as nn
import torch

from .module_queue import ModuleQueue

class VGGClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGGClassifier, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

VGG_LAYERS = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(ModuleQueue):
    def __init__(self, variant="A", start=0, end=-1, num_classes=1000, state_path = None, init_weights=False, wrap_layers=False):
        if wrap_layers:
            partitions = nn.Sequential(*make_layers(variant, start, end))
        else:
            partitions = make_layers(variant, start, end)
        if end < 0 or end > len(VGG_LAYERS[variant]):
            partitions.append(VGGClassifier(num_classes))
        ModuleQueue.__init__(self, partitions, start)

        if not self.load_parameters(state_path, "VGG"):
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(variant, start, end, batch_norm=True):
    layers = []
    in_channels = 3
    num_layers = end - start
    traversed_layers = 0
    for v in VGG_LAYERS[variant]:
        if v == 'M':
            new_layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                new_layers = [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                new_layers = [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

        if traversed_layers >= start:
            layers += new_layers

        if end > 0 and len(layers) >= num_layers:
            break
        else:
            traversed_layers += len(new_layers)

    return layers

class VGG_unsplit(VGG):
    def __init__(self, **args):
        super().__init__(**args)

class VGG_server(VGG):
    def __init__(self, **args):
        super().__init__(start = 8, **args)

class VGG_client(VGG):
    def __init__(self, **args):
        super().__init__(end = 8, **args)
