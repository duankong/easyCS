import numpy as np
from model import UNet, NestedUNet, UNet_res, UNet_conv, ESS_net, DQBCS


def get_model(model, n_channels, n_classes):
    global net
    if model == "Unet":
        net = UNet(n_channels=n_channels, n_classes=n_classes)
    elif model == "nestedUnet":
        net = NestedUNet(n_channels=n_channels, n_classes=n_classes, deepsupervision=False)
    elif model == "Unet_res":
        net = UNet_res(n_channels=n_channels, n_classes=n_classes)
    elif model == "Unet_conv":
        net = UNet_conv(n_channels=n_channels, n_classes=n_classes, bilinear=True, Measure_return=False)
    elif model == "Essnet":
        net = ESS_net(in_channels=n_channels, out_channels=n_classes)
    elif model == "DQBCS":
        net = DQBCS(in_channels=n_channels, out_channels=n_classes)
    else:
        return 0
    return net

def get_time(time):
    time=np.array(time).astype(np.int64)
    # print(time)
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    time_log="{:02d}:{:02d}:{:02d}".format(h,m,s)
    return time_log


if __name__ == '__main__':
    time=6666666.3
    print(get_time(time))
