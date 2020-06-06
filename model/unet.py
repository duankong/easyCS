import torch
import torch.nn as nn
import torch.nn.functional as F
from config import args_config

args_ = args_config()

if args_.test_model == True:
    num_feature = 2
else:
    num_feature = 64


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, num_feature)
        self.down1 = Down(num_feature, num_feature * 2)
        self.down2 = Down(num_feature * 2, num_feature * 4)
        self.down3 = Down(num_feature * 4, num_feature * 8)
        self.down4 = Down(num_feature * 8, num_feature * 8)
        self.up1 = Up(num_feature * 16, num_feature * 4, bilinear)
        self.up2 = Up(num_feature * 8, num_feature * 2, bilinear)
        self.up3 = Up(num_feature * 4, num_feature, bilinear)
        self.up4 = Up(num_feature * 2, num_feature, bilinear)
        self.outc = OutConv(num_feature, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out


class UNet_res(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_res, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, num_feature)
        self.down1 = Down(num_feature, num_feature * 2)
        self.down2 = Down(num_feature * 2, num_feature * 4)
        self.down3 = Down(num_feature * 4, num_feature * 8)
        self.down4 = Down(num_feature * 8, num_feature * 8)
        self.up1 = Up(num_feature * 16, num_feature * 4, bilinear)
        self.up2 = Up(num_feature * 8, num_feature * 2, bilinear)
        self.up3 = Up(num_feature * 4, num_feature, bilinear)
        self.up4 = Up(num_feature * 2, num_feature, bilinear)
        self.final = nn.Conv2d(num_feature, n_channels, kernel_size=1)
        self.Sigmod = nn.Sigmoid()

    def forward(self, x):
        input = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = torch.add(self.final(x), input)
        out = self.Sigmod(out)
        return out


class UNet_conv(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True, Measure_return=False):
        super(UNet_conv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.Measure_return = Measure_return
        self.Measure = Measure_net(n_channels, n_channels)
        self.inc = DoubleConv(n_channels, num_feature)
        self.down1 = Down(num_feature, num_feature * 2)
        self.down2 = Down(num_feature * 2, num_feature * 4)
        self.down3 = Down(num_feature * 4, num_feature * 8)
        self.down4 = Down(num_feature * 8, num_feature * 8)
        self.up1 = Up(num_feature * 16, num_feature * 4, bilinear)
        self.up2 = Up(num_feature * 8, num_feature * 2, bilinear)
        self.up3 = Up(num_feature * 4, num_feature, bilinear)
        self.up4 = Up(num_feature * 2, num_feature, bilinear)
        self.outc = OutConv(num_feature, n_classes)

    def forward(self, x):
        m_img, p_img = self.Measure(x)
        x1 = self.inc(p_img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)

        if self.Measure_return:
            return out, m_img
        else:
            return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmod(x)
        return x


class Measure_net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Measure_net, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, padding=2, dilation=2, kernel_size=3, stride=5)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, padding=2, dilation=2,
                                     kernel_size=3, stride=5)

    def forward(self, x):
        m_img = self.down(x)
        pre_img = self.up(m_img)
        return m_img, pre_img
