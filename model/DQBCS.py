# An efficient deep quantized compressed sensing coding framework of natural images
# Cui, Wenxue, et al. “An Efficient Deep Quantized Compressed Sensing Coding Framework of Natural Images.” MM 2018 - Proceedings of the 2018 ACM Multimedia Conference, 2018, pp. 1777–85, doi:10.1145/3240508.3240706.
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import args_config

B = 32


class Sample_subNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Sample_subNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=B, stride=B)

    def forward(self, input):
        x = self.conv(input)
        return x

class Offset_subNetwork(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Offset_subNetwork,self).__init__()

    def forward(self, input):
        return None


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
