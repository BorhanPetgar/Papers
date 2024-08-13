"""
This is implementation of `SigNet: Convolutional Siamese Network for Writer Independent Offline Signature
Verification` paper.
The color of the layer is based on the color of the layer in the paper.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class SigNet(nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()
        self.blue1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True)
        )
        self.magneta1 = nn.Sequential(
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
        )
        self.yellow1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.green1 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.magneta2 = nn.Sequential(
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        )
        self.yellow2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.grey1 = nn.Sequential(
            nn.Dropout2d(p=0.3)
        )
        self.cyan1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.cyan2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.yellow3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.grey2 = nn.Sequential(
            nn.Dropout2d(p=0.3)
        )

    def forward_branch(self, x):
        x = self.blue1(x)
        x = self.magneta1(x)
        x = self.yellow1(x)
        x = self.green1(x)
        x = self.magneta2(x)
        x = self.yellow2(x)
        x = self.grey1(x)
        x = self.cyan1(x)
        x = self.cyan2(x)
        x = self.yellow3(x)
        x = self.grey2(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_branch(x1)
        out2 = self.forward_branch(x2)
        return out1, out2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        contrastive_loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2)) + \
                            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0), 2)
        return contrastive_loss

model = SigNet()
print(model(torch.randn(1, 1, 155, 220), torch.randn(1, 1, 155, 220)).shape)
