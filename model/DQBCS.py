# An efficient deep quantized compressed sensing coding framework of natural images
# Cui, Wenxue, et al. “An Efficient Deep Quantized Compressed Sensing Coding Framework of Natural Images.” MM 2018 - Proceedings of the 2018 ACM Multimedia Conference, 2018, pp. 1777–85, doi:10.1145/3240508.3240706.
import torch
import torch.nn as nn
import numpy as np

from torchsummary import summary

B = 32
rate = 0.02
Nb = int(np.floor(rate * B * B))
Recon_filter = 64
step = 12


class DQBCS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DQBCS, self).__init__()
        self.Sample = Sample_subNetwork(in_channels, Nb)
        self.Offset = Offset_subNetwork(Nb, Nb)
        self.Reconstruction = Reconstruction_subNetwork(Nb, out_channels)

    def forward(self, input):
        measure = self.Sample(input)
        quantized = measure / step

        offset = self.Offset(quantized) + step
        dequantized = quantized * offset
        output = self.Reconstruction(dequantized)

        return output


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
        self.conv0 = nn.Conv2d(in_channels, out_channels=B * B, kernel_size=1)
        self.Shuffle = nn.PixelShuffle(B)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=Recon_filter, kernel_size=7, padding=3)
        self.RB = ResBlock(in_channels=Recon_filter, out_channels=Recon_filter)
        self.conv2 = nn.Conv2d(in_channels=Recon_filter, out_channels=out_channels, kernel_size=7, padding=3)

    def forward(self, input):
        x = self.conv0(input)
        x = self.Shuffle(x)
        x = self.conv1(x)
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
    in_channel = 1
    out_channel = 1
    width = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    my_net = DQBCS(in_channels=in_channel, out_channels=out_channel).to(device)

    summary(my_net, input_size=(in_channel, width, width))

    img = torch.rand(1, in_channel, width, width).to(device)

    result = my_net(img)
    print('input shape is : {}'.format(img.shape))
    print('out shape is :{}'.format(result.shape))
