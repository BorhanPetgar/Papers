import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.down_sample = down_sample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.down_sample is not None:
            identity = self.down_sample
        
        out += identity
        out = self.relu(out)
        

class Resnet18(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        
    def forward(self, x):
        pass