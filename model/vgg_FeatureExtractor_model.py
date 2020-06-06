import torch.nn as nn
import torch
from torchvision.models import vgg16
import numpy as np
import skimage.measure
import scipy


def vgg_prepro(x):
    x = x.resize([244, 244], interp='bilinear', mode=None)
    # x = np.tile(x, 3)
    x = x / 127.5 - 1
    return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg16_model = vgg16(pretrained=True)
        self.downsample = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.vgg16_54 = nn.Sequential(*list(vgg16_model.features.children())[:22])
        self.output = nn.Sequential(nn.Conv2d(512, 1, kernel_size=3, padding=1))

    def forward(self, img):
        img = self.downsample(img)
        img = img * 255 / 127.5 - 1
        img = self.conv1(img)
        img = self.vgg16_54(img)
        img = self.output(img)
        img = img.view(-1)
        return img


if __name__ == '__main__':
    model = FeatureExtractor()
    input_224 = torch.rand(1, 1, 512, 512)
    print(input_224.size())  # (1,3,224,224)
    output = model(input_224)
    print(output.size())  # (1,2048,1,1)
