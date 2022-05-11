import torch
import torch.nn as nn


class encodingUnit(nn.Module):
    def __init__(self, compressionProps, prevLayerProps):
        super(encodingUnit, self).__init__()

        self.compressionProps = compressionProps

        prevLayerChannels = prevLayerProps["PrevLayerOutChannel"]
        compressedChannelNum = int(
            prevLayerChannels / compressionProps["feature_compression_factor"]
        )

        kernelHeight = 3
        kernelWidth = 3
        strideVal = 1
        paddingVal = 1

        self.convIn1 = nn.Conv2d(
            in_channels=prevLayerChannels,
            out_channels=compressedChannelNum,
            kernel_size=(kernelHeight, kernelWidth),
            stride=strideVal,
            padding=paddingVal,
        )
        self.batchNormIn = nn.BatchNorm2d(compressedChannelNum, momentum=0.03, eps=1e-4)

    def forward(self, x):
        x = torch.sigmoid(self.batchNormIn(self.convIn1(x)))
        return x


class decodingUnit(nn.Module):
    def __init__(self, compressionProps, prevLayerProps):
        super(decodingUnit, self).__init__()

        self.compressionProps = compressionProps

        prevLayerChannels = prevLayerProps["PrevLayerOutChannel"]
        compressedChannelNum = int(
            prevLayerChannels / compressionProps["feature_compression_factor"]
        )

        kernelHeight = 3
        kernelWidth = 3
        strideVal = 1
        paddingVal = 1

        self.convOut1 = nn.ConvTranspose2d(
            in_channels=compressedChannelNum,
            out_channels=prevLayerChannels,
            kernel_size=(kernelHeight, kernelWidth),
            stride=strideVal,
            padding=paddingVal,
        )

    def forward(self, x):
        x = torch.relu(self.convOut1(x))
        return x
