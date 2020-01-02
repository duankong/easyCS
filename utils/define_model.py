from model import unet, nestedUnet


def get_model(model_name, n_channels, n_classes):
    if model_name == "Unet":
        net = unet.UNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name == "nestedUnet":
        net = nestedUnet.NestedUNet(n_channels=n_channels, n_classes=n_classes, deepsupervision=False)
    return net
