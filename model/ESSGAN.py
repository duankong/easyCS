import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

fnum = 6


class ESS_net(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ESS_net, self).__init__()

    def forward(self, x):
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_i = PurpleConv(in_channels, fnum, stride=2)
        self.RIRB_block = RIRB(fnum, fnum)
        self.conv_o = PurpleConv(2 * fnum, out_channels, stride=1)

    def forward(self, x):
        x1 = self.conv_i(x)
        x2 = self.RIRB_block(x1)
        x3 = self.conv_o(x2)
        return x3


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_i = OrangeConv(in_channels, fnum, stride=1)
        self.RIRB_block = RIRB(fnum, fnum)
        self.conv_o = OrangeConv(2 * fnum, out_channels, stride=2)

    def forward(self, x):
        x1 = self.conv_i(x)
        x2 = self.RIRB_block(x1)
        x3 = self.conv_o(x2)
        return x3


class RIRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        midfnum = int(fnum / 2)
        self.input = RedConv(in_channels, midfnum)
        self.CBL_mid1 = RedConv(midfnum, midfnum, kernel_size=1, padding=0)
        self.output = RedConv(fnum, fnum)

    def forward(self, x):
        cbl1 = self.input(x)
        cbl2 = self.CBL_mid1(cbl1)
        cbl3 = self.CBL_mid1(cbl2)
        cbl4 = self.output(torch.cat([cbl3, cbl1], dim=1))
        out = torch.cat([cbl4, x], dim=1)
        return out


class RedConv(nn.Module):
    # """(convolution => [BN] => LeakyReLU) """ the same

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.red_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.red_conv(x)


class PurpleConv(nn.Module):
    # """(convolution => [BN] => LeakyReLU) """ down sample

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.Purple_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.Purple_conv(x)


class OrangeConv(nn.Module):
    # """(convolution => [BN] => LeakyReLU) """ up sample

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.orange_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.orange_conv(x)


if __name__ == '__main__':
    my_net = DecoderBlock(1, 2)
    summary(my_net, input_size=(1, 256, 256))
    img = torch.rand(1, 1, 256, 256)
    result = my_net(img)
    print(result.shape)
