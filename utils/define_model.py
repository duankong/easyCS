from model import UNet, NestedUNet, UNet_res, UNet_conv, ESS_net


def get_model(model_name, n_channels, n_classes):
    global net
    if model_name == "Unet":
        net = UNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name == "nestedUnet":
        net = NestedUNet(n_channels=n_channels, n_classes=n_classes, deepsupervision=False)
    elif model_name == "Unet_res":
        net = UNet_res(n_channels=n_channels, n_classes=n_classes)
    elif model_name == "Unet_conv":
        net = UNet_conv(n_channels=n_channels, n_classes=n_classes, bilinear=True, Measure_return=False)
    elif model_name == "Essnet":
        net = ESS_net(in_channels=n_channels, out_channels=n_classes)
    else:
        return 0
    return net
