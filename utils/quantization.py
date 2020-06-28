# 二维图像的离散变余弦换（DCT）
# Python3.5
# 库：cv2+numpy+matplotlib
# 作者：James_Ray_Murphy
import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft, ifftshift

from utils import get_random_0_1_left_right, get_random_0_1_centre, scale_img


def dct_comp(img, mycent):
    array_dct = get_random_0_1_left_right(img.shape[0], img.shape[1], mycent)
    img_dct = cv2.dct(img) * array_dct  # 进行离散余弦变换
    img_dct_log = np.log(abs(cv2.dct(img))) * array_dct  # 进行log处理
    dct_record = cv2.idct(img_dct)  # 进行离散余弦反变换
    return img_dct_log, dct_record


def fft_cmp(img, mycent):
    array_fft = get_random_0_1_centre(img.shape[0], img.shape[1], mycent)
    fft0 = scipy.fftpack.fft2(img)
    fft0 = scipy.fftpack.fftshift(fft0)
    fft = fft0 * array_fft

    fft_show = np.log(abs(fft0)) * array_fft  # 进行log处理
    ifft = scipy.fftpack.ifftshift(fft)
    fft_record = scipy.fftpack.ifft2(ifft)
    fft_record = np.abs(fft_record)
    return fft_show, fft_record


def Uniform_quantization(img, delta):
    uniform_img = np.int(img / delta)
    restore_img = uniform_img * delta
    return uniform_img, restore_img


def show_compress_data(img, dct_log, dct_record, fft_show, fft_record):
    # PLOT
    plt.figure()
    plt.subplot(231), plt.imshow(img, 'gray'), plt.title('original image'), plt.axis('off')
    plt.subplot(232), plt.imshow(fft_show, 'gray'), plt.title('FFT'), plt.axis('off')
    plt.subplot(233), plt.imshow(fft_record, 'gray'), plt.title('IFFT')

    plt.subplot(234), plt.imshow(img, 'gray'), plt.title('original image'), plt.axis('off')
    plt.subplot(235), plt.imshow(dct_log, 'gray'), plt.title('DCT'), plt.axis('off')
    plt.subplot(236), plt.imshow(dct_record, 'gray'), plt.title('IDCT')

    plt.show()


if __name__ == '__main__':
    # data
    img = cv2.imread('E:/Desktop/easyCS/data/limotiff/6.tif', 0)
    img = img.astype('float')

    # img = cv2.imread("E:/Desktop/easyCS/data/limotiff/lena.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sizeisame = True

    if sizeisame == True:
        sizeinfo = [256]
        mycent_num = [0.06, 0.05, 0.04]
    else:
        sizeinfo = [512, 384, 256, 192, 128, 64]
        mycent_num = [0.05]
    # para

    # show
    for i, size_ in enumerate(sizeinfo):
        for j, mycent_ in enumerate(mycent_num):
            print(size_)
            imgResize = scale_img(img, size=size_)
            dct_log, dct_record = dct_comp(img=img, mycent=mycent_)
            fft_show, fft_record = fft_cmp(img=img, mycent=mycent_)

            show_compress_data(img, dct_log, dct_record, fft_show, fft_record)
    # print('[*] 百分比是 {:.2f}'.format(mycent_num * 100))
    print('[*] Done')
