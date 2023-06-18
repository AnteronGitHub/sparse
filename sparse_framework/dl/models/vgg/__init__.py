import torch.nn as nn
import torch


from .compression_utils_vgg import encodingUnit
from .compression_utils_vgg import decodingUnit

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False, local = False, use_compression = False):
        super(VGG, self).__init__()
        self.use_compression = use_compression
        self.local = local
        self.encoder = []
        self.decoder = []
        if use_compression and local:
            self.features, self.encoder = features
        else: 
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

    '''
    def forward(self, x, local = False):
        x = self.features(x)
        if local == False:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x
    
    '''
    
    def forward(self, x):
        if self.local == False:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
        else:
            if self.use_compression:
                x, self.prune_filter = self.features(x)
                return x, self.prune_filter
            else:
                x = self.features(x)
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
            
    def resetPrune(self):
        self.encoder.resetPrune()
        
    def resetdePrune(self):
        self.encoder.resetdePrune()
        

def resetPrune(model):
    model[-1].resetPrune()


def make_layers(cfg, compressionProps=None, in_channels=3, batch_norm=True, local=False, use_compression=False):
    layers = []

    if use_compression and not local:
        # Add decompression layer
        prevLayerProps = {}
        prevLayerProps["PrevLayerOutChannel"] = in_channels
        layers += [decodingUnit(compressionProps,prevLayerProps)]
        in_channels = in_channels

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

    if use_compression and local:
        # Add compression layer
        prevLayerProps = {}
        prevLayerProps["PrevLayerOutChannel"] = in_channels
        encoder = encodingUnit(compressionProps,prevLayerProps)
        layers += [encoder]
        return nn.Sequential(*layers), encoder
    else:
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
    def __init__(self, compressionProps, use_compression):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device, is server")
        super().__init__(make_layers(cfg_server["A"],
                                     compressionProps,
                                     in_channels=128,
                                     local = False,
                                     use_compression=use_compression),
                         local = False,
                         use_compression=use_compression)

class VGG_client(VGG): #in_channel is the num filters in previous channel
    def __init__(self, compressionProps, use_compression):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device, is client")
        super().__init__(make_layers(cfg_local["A"],
                                     compressionProps,
                                     local = True,
                                     use_compression=use_compression),
                         local = True,
                         use_compression=use_compression)
