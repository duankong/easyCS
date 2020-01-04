from model import unet, nestedUnet


def get_model(model_name, n_channels, n_classes):
    global net
    if model_name == "Unet":
        net = unet.UNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name == "nestedUnet":
        net = nestedUnet.NestedUNet(n_channels=n_channels, n_classes=n_classes, deepsupervision=False)
    elif model_name == "Unet_res":
        net = unet.UNet_res(n_channels=n_channels, n_classes=n_classes)
    else:
        return 0
    return net
