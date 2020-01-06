import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
from data import get_mask


def DWT_demo():
    img = cv2.imread("../data/lena.jpg")
    img = pywt.data.aero()

    # img = cv2.resize(img, (448, 448))
    # 将多通道图像变为单通道图像
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    plt.figure('二维小波一级变换')
    coeffs = pywt.dwt2(img, 'db1')
    LL, (LH, HL, HH) = coeffs

    # 将各个子图进行拼接，最后得到一张图
    AH = np.concatenate([LL, LH], axis=1)
    VD = np.concatenate([HL, HH], axis=1)
    result = np.concatenate([AH, VD], axis=0)

    img_all = list()
    img_all.append(img)
    img_all.append(LL)
    img_all.append(LH)
    img_all.append(HL)
    img_all.append(HH)
    img_all.append(result)
    for step, img_ in enumerate(img_all):
        print('[@] {:02d}'.format(step + 1))
        plt.figure('2')
        plt.imshow(img_, 'gray')
        plt.show()


def DCT_demo():
    img = pywt.data.aero().astype(np.float)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img_dct = cv2.dct(img)  # 进行离散余弦变换
    img_dct_log = np.log(abs(img_dct))  # 进行log处理
    img_recor = cv2.idct(img_dct)  # 进行离散余弦反变换

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.title('Original img')
    plt.imshow(img, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title('full dct')
    plt.imshow(img_dct, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('full dct log')
    plt.imshow(img_dct_log, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title('recon')
    plt.imshow(img_recor, cmap='gray')

    plt.show()


if __name__ == '__main__':
    DCT_demo()
