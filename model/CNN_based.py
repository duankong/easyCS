import torch.nn as nn
import torch


class CNN_based(nn.Module):

    def __init__(self, n_channels):
        super(CNN_based, self).__init__()
        df_dim = 1
        self.cb0 = Conv_Batch(n_channels, df_dim)
        self.cb1 = Conv_Batch(df_dim, df_dim * 2)
        self.cb2 = Conv_Batch(df_dim * 2, df_dim * 4)
        self.cb3 = Conv_Batch(df_dim * 4, df_dim * 8)
        self.cb4 = Conv_Batch(df_dim * 8, df_dim * 16)
        self.cb5 = Conv_Batch(df_dim * 16, df_dim * 32)
        self.cb6 = Conv_Batch(df_dim * 32, df_dim * 16)
        self.cb7 = Conv_Batch(df_dim * 16, df_dim * 8)
        self.res_cd7 = net_res(df_dim * 8, df_dim * 8)
        self.out = out_put()

    def forward(self, x1):
        x1 = self.cb0(x1)
        x1 = self.cb1(x1)
        x1 = self.cb2(x1)
        x1 = self.cb3(x1)
        x1 = self.cb4(x1)
        x1 = self.cb5(x1)
        x1 = self.cb6(x1)
        x1 = self.cb7(x1)

        x2 = self.res_cd7(x1)

        x3 = torch.cat([x2, x1], dim=1)

        x3 = x3.view(-1, 16 * 2 * 2)

        x3 = self.out(x3)
        return x3


class out_put(nn.Module):
    def __init__(self):
        super().__init__()
        self.outputc = nn.Sequential(
            nn.Linear(16 * 2 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.outputc(x)


class Conv_Batch(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.my_Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.my_Conv(x)


class net_res(nn.Module):
    """"res"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_out = int(out_channels / 4)

        self.res_struct = nn.Sequential(

            nn.Conv2d(in_channels, num_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_out, num_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_out, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.res_struct(x)
