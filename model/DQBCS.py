# An efficient deep quantized compressed sensing coding framework of natural images
# Cui, Wenxue, et al. “An Efficient Deep Quantized Compressed Sensing Coding Framework of Natural Images.” MM 2018 - Proceedings of the 2018 ACM Multimedia Conference, 2018, pp. 1777–85, doi:10.1145/3240508.3240706.
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import args_config

B = 32
Nb = 6
Recon_filter = 64




class Sample_subNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Sample_subNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=B, stride=B)

    def forward(self, input):
        x = self.conv(input)
        return x


class Offset_subNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Offset_subNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=Nb, kernel_size=kernel_size, stride=stride, padding=1)
        self.RB = ResBlock(in_channels=Nb, out_channels=out_channels)

    def forward(self, input):
        x = input
        x1 = self.conv1(x)
        rb1 = self.RB(x1)
        rb2 = self.RB(rb1)
        rb3 = self.RB(rb2)
        rb4 = self.RB(rb3)
        x2 = self.conv1(rb4)
        output = x2 + input
        return output


class Reconstruction_subNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(Reconstruction_subNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=Recon_filter, kernel_size=7)
        self.RB = ResBlock(in_channels=Recon_filter, out_channels=Recon_filter)
        self.conv2 = nn.Conv2d(in_channels=Recon_filter, out_channels=out_channels, kernel_size=7)

    def forward(self, input):
        x = self.conv1(input)
        for i in range(5):
            x = self.RB(x)
        output = self.conv2(x)
        return output


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.Relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.bn(x)

        x = self.Relu(x)

        x = self.conv1(x)
        x = self.bn(x)

        x = x + input
        output = self.Relu(x)
        return output


if __name__ == '__main__':
    pass
