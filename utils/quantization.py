# 二维图像的离散变余弦换（DCT）
# Python3.5
# 库：cv2+numpy+matplotlib
# 作者：James_Ray_Murphy
import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft, ifftshift

from utils import get_random_0_1_left_right,get_random_0_1_centre,scale_img


def show_compress_data(mycent, img):
    myarrayfft = get_random_0_1_centre(img.shape[0], img.shape[1], mycent)
    myarraydct = get_random_0_1_left_right(img.shape[0], img.shape[1], mycent)
    # myarraydwt = get_random_0_1_left_right(img.shape[0], img.shape[1], mycent)
    # FFT
    fft0 = scipy.fftpack.fft2(img)
    fft0 = scipy.fftpack.fftshift(fft0)
    fft = fft0 * myarrayfft
    fft_show = np.log(abs(fft0)) * myarrayfft  # 进行log处理
    # fft_show[fft_show <= 0] = 255
    ifft = scipy.fftpack.ifftshift(fft)
    img_recor1 = scipy.fftpack.ifft2(ifft)
    img_recor1 = np.abs(img_recor1)
    # DCT
    img_dct = cv2.dct(img) * myarraydct  # 进行离散余弦变换
    img_dct_log = np.log(abs(cv2.dct(img))) * myarraydct  # 进行log处理
    # img_dct_log[img_dct_log <= 0] = 255
    img_recor2 = cv2.idct(img_dct)  # 进行离散余弦反变换
    # DWT
    # coeffs = pywt.wavedecn(img, 'haar', level=2)
    # arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    # coeffs_from_arr = pywt.array_to_coeffs(arr * myarraydwt, coeff_slices, output_format='wavedecn')
    # arr_show = (arr) * myarraydwt
    # # arr_show[arr_show <= 0] = 255
    # idwt = pywt.waverecn(coeffs_from_arr, 'haar')

    # PLOT
    plt.figure()
    plt.subplot(231), plt.imshow(img, 'gray'), plt.title('original image'), plt.axis('off')
    plt.subplot(232), plt.imshow(fft_show, 'gray'), plt.title('FFT'), plt.axis('off')
    plt.subplot(233), plt.imshow(img_recor1, 'gray'), plt.title('IFFT')

    plt.subplot(234), plt.imshow(img, 'gray'), plt.title('original image'), plt.axis('off')
    plt.subplot(235), plt.imshow(img_dct_log, 'gray'), plt.title('DCT'), plt.axis('off')
    plt.subplot(236), plt.imshow(img_recor2, 'gray'), plt.title('IDCT')

    # plt.subplot(337), plt.imshow(img, 'gray'), plt.title('original image')
    # plt.subplot(338), plt.imshow(arr_show, 'gray'), plt.title('DWT')
    # plt.subplot(339), plt.imshow(idwt, 'gray'), plt.title('IDWT')

    plt.show()


if __name__ == '__main__':
    # data
    img = cv2.imread('E:/Desktop/easyCS/data/limotiff/6.tif', 0)
    img = img.astype('float')

    # img = cv2.imread("E:/Desktop/easyCS/data/limotiff/lena.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sizeissame = True

    if sizeissame == True:
        sizeinfo = [128]
        mycent_num = [0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    else:
        sizeinfo = [512, 384, 256, 192, 128, 64]
        mycent_num = [0.05]
    # para

    # show
    for i, size_ in enumerate(sizeinfo):
        for j, mycent_ in enumerate(mycent_num):
            print(size_)
            imgResize = scale_img(img, size=size_)
            show_compress_data(img=imgResize, mycent=mycent_)

    # print('[*] 百分比是 {:.2f}'.format(mycent_num * 100))
    print('[*] Done')
