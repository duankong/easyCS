import torch
import torch.nn as nn
import torch.nn.functional as F
from config import args_config

args_ = args_config()
if args_.test_model == True:
    fnum = 4
else:
    fnum = 64


class ESS_net(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ESS_net, self).__init__()
        self.input = BlueCircle(in_channels, fnum)
        self.down = EncoderBlock(fnum, fnum)
        self.up = DecoderBlock(fnum, fnum)
        self.rirb = RIRB(fnum, fnum)
        self.output = MagentaCircle(fnum, out_channels)

    def forward(self, x):
        d0_0 = self.input(x)
        d0_1 = self.down(d0_0)
        d0_2 = self.down(d0_1)
        d0_3 = self.down(d0_2)
        d0_4 = self.down(d0_3)
        u0_3 = self.up(d0_4) + self.rirb(d0_3)
        u0_2 = self.up(u0_3) + self.rirb(d0_2)
        u0_1 = self.up(u0_2) + self.rirb(d0_1)
        u0_0 = self.up(u0_1) + self.rirb(d0_0)
        x1 = self.output(u0_0)

        d1_0 = self.input(x1) + self.rirb(u0_0)
        d1_1 = self.down(d1_0) + self.rirb(u0_1)
        d1_2 = self.down(d1_1) + self.rirb(u0_2)
        d1_3 = self.down(d1_2) + self.rirb(u0_3)
        d1_4 = self.down(d1_3) + self.rirb(d0_4)

        u1_3 = self.up(d1_4) + self.rirb(d1_3)
        u1_2 = self.up(u1_3) + self.rirb(d1_2)
        u1_1 = self.up(u1_2) + self.rirb(d1_1)
        u1_0 = self.up(u1_1) + self.rirb(d1_0)

        out = self.output(u1_0)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels=fnum, out_channels=fnum):
        super().__init__()

        self.conv_i = PurpleConv(in_channels, fnum, stride=2)
        self.RIRB_block = RIRB(fnum, fnum)
        self.conv_o = PurpleConv(fnum, out_channels, stride=1)

    def forward(self, x):
        x1 = self.conv_i(x)
        x2 = self.RIRB_block(x1)
        x3 = self.conv_o(x2)
        return x3


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_i = OrangeConv(in_channels, fnum, stride=1, padding=1)
        self.RIRB_block = RIRB(fnum, fnum)
        self.conv_o = OrangeConv(fnum, out_channels, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x1 = self.conv_i(x)
        x2 = self.RIRB_block(x1)
        x3 = self.conv_o(x2)
        return x3


class RIRB(nn.Module):
    def __init__(self, in_channels=fnum, out_channels=fnum):
        super().__init__()
        midfnum = int(fnum / 2)
        self.input = RedConv(in_channels, midfnum)
        self.CBL_mid1 = RedConv(midfnum, midfnum, kernel_size=1, padding=0)
        self.output = RedConv(midfnum, fnum)

    def forward(self, x):
        cbl1 = self.input(x)
        cbl2 = self.CBL_mid1(cbl1)
        cbl3 = self.CBL_mid1(cbl2)
        cbl4 = self.output(torch.add(cbl3, cbl1))
        out = cbl4 + x
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

    def __init__(self, in_channels, out_channels, stride=2, output_padding=0, padding=1):
        super().__init__()
        self.orange_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=stride,
                               output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.orange_conv(x)


class BlueCircle(nn.Module):
    # """(convolution => [BN] => LeakyReLU) """ blue circle for input
    def __init__(self, in_channels, out_channels):
        super(BlueCircle, self).__init__()
        self.blue_circle = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.blue_circle(x)


class MagentaCircle(nn.Module):
    # """(convolution => [BN] => LeakyReLU) """ blue circle for output
    def __init__(self, in_channels, out_channels):
        super(MagentaCircle, self).__init__()
        self.magenta_circle = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.magenta_circle(x)


if __name__ == '__main__':
    in_channel = 1
    out_channel = 1
    width = 256
    my_net = ESS_net(in_channels=in_channel, out_channels=out_channel)
    img = torch.rand(1, in_channel, width, width)
    result = my_net(img)
    print('input shape is : {}'.format(img.shape))
    print('out shape is :{}'.format(result.shape))
