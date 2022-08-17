from calendar import c
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch


from models.compression_utils import encodingUnit
from models.compression_utils import decodingUnit


class VGG(nn.Module):
    def __init__(self, features, local = False, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.local = local
        self.features = features
        if self.local == False:
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

    """
    def forward(self, x, local = False):
        x = self.features(x)
        if local == False:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x
    
    """

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
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, compressionProps=None, Prev_in_channels=None, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "CL":
            prevLayerProps = {}
            prevLayerProps["PrevLayerOutChannel"] = in_channels
            layers += [encodingUnit(compressionProps, prevLayerProps)]
        elif v == "CS":
            prevLayerProps = {}
            prevLayerProps["PrevLayerOutChannel"] = Prev_in_channels
            layers += [decodingUnit(compressionProps, prevLayerProps)]
            in_channels = Prev_in_channels
        elif v == "M":
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
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}
cfg_local = {
    "A": [
        64,
        "M",
        128,
        "M",
        "CL"],
    "E": [64, 64, "M", 128, 128, "CL"],
}
cfg_server = {
    "A": ["CS",256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "E": [
        "CS",
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}  #CS, CL active


class VGG_unsplit(VGG):
    def __init__(self, **kwargs):
        super().__init__(make_layers(cfg["A"]), **kwargs)

class VGG_server(VGG):
    def __init__(self,
                 feature_compression_factor: int = 1,
                 resolution_compression_factor: int = 1,
                 **kwargs):
        compression_props = {
            "feature_compression_factor": feature_compression_factor,
            "resolution_compression_factor": resolution_compression_factor,
        }
        super().__init__(make_layers(cfg_server["A"], compression_props, Prev_in_channels=128), local = False, **kwargs)

class VGG_client(VGG):
    def __init__(self,
                 feature_compression_factor: int = 1,
                 resolution_compression_factor: int = 1,
                 **kwargs):
        compression_props = {
            "feature_compression_factor": feature_compression_factor,
            "resolution_compression_factor": resolution_compression_factor,
        }
        super().__init__(make_layers(cfg_local["A"], compression_props), local = True, **kwargs)

