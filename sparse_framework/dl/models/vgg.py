import torch.nn as nn
import torch


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False, local = False):
        super(VGG, self).__init__()
        self.local = local
        self.features = features
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
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)

        if self.local == False:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)

        return x

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

def make_layers(cfg, in_channels=3, batch_norm=True):
    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
cfg_local = {
    #'A': [64, 'M', 128, 'M'],
    #'E': [64, 64, 'M', 128, 128],
    'A': [64, 'M', 128, 'M'],
    'D': [64],
    'E': [64, 64, 'M', 128, 128],
}
cfg_server = {
    #'A': [256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    #'E': ['M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'A': [256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': ['M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_unsplit(VGG):
    def __init__(self):
        super().__init__(make_layers(cfg["A"]))

class VGG_server(VGG):
    def __init__(self):
        super().__init__(make_layers(cfg_server["A"],
                                     in_channels=128),
                         local = False)

class VGG_client(VGG):
    def __init__(self):
        super().__init__(make_layers(cfg_local["A"]),
                         local = True)
