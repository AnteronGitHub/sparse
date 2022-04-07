import random
random.seed(57)

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F # import convolution functions like Relu
from models.compressor import encodingUnit
from models.compressor import decodingUnit

# Define a convolution neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*8*8, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 24*8*8)
        output = self.fc1(output)

        return output
    
class NeuralNetwork_local(nn.Module):
    def __init__(self, compressionProps):
        super(NeuralNetwork_local, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        
        prevLayerProps = {}
        prevLayerProps["PrevLayerOutChannel"] = 12
        
        self.encoder = encodingUnit(compressionProps,prevLayerProps)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))      
        x = F.relu(self.bn2(self.conv2(x)))     
        x = self.pool(x)   
        x = self.encoder(x)                   

        return x
    
class NeuralNetwork_server(nn.Module):
    def __init__(self, compressionProps):
        super(NeuralNetwork_server, self).__init__()
        
        prevLayerProps = {}
        prevLayerProps["PrevLayerOutChannel"] = 12
        
        self.decoder = decodingUnit(compressionProps,prevLayerProps)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*8*8, 10)

    def forward(self, x):   
        
        x = self.decoder(x)                
        x = F.relu(self.bn4(self.conv4(x)))     
        x = F.relu(self.bn5(self.conv5(x)))     
        x = x.view(-1, 24*8*8)
        x = self.fc1(x)
  

        return x