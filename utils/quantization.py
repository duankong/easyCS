# 二维图像的离散变余弦换（DCT）
# Python3.5
# 库：cv2+numpy+matplotlib
# 作者：James_Ray_Murphy
import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft, ifftshift


def get_random_0_1(wide, high, percent=0.01):
    nums = np.zeros(wide * high)
    nums[:int(wide * high * percent)] = 1
    np.random.shuffle(nums)
    nums = nums.reshape(wide, high)
    return nums


def get_random_0_1_left_right(wide, high, len):
    nums = np.zeros([wide, high])
    nums[0:len, 0:len] = 1
    return nums


def get_random_0_1_centre(wide, high, len):
    nums = np.zeros([wide, high])
    a = int(wide / 2) - int(len / 2)
    b = int(wide / 2) + int(len / 2)
    c = int(high / 2) - int(len / 2)
    d = int(high / 2) + int(len / 2)

    nums[a:b, c:d] = 1
    return nums


def show_compress_data(mycent, img):
    myarrayfft = get_random_0_1_centre(img.shape[0], img.shape[1], mycent)
    myarraydct = get_random_0_1_left_right(img.shape[0], img.shape[1], mycent)
    # FFT
    fft0 = scipy.fftpack.fft2(img)
    fft0 = scipy.fftpack.fftshift(fft0)
    fft = fft0 * myarrayfft
    fft_show = np.log(abs(fft))  # 进行log处理
    ifft = scipy.fftpack.ifftshift(fft)
    img_recor1 = scipy.fftpack.ifft2(ifft)
    img_recor1 = np.abs(img_recor1)
    # DCT
    img_dct = cv2.dct(img) * myarraydct  # 进行离散余弦变换
    img_dct_log = np.log(abs(img_dct))  # 进行log处理
    img_recor2 = cv2.idct(img_dct)  # 进行离散余弦反变换
    # PLOT
    plt.subplot(231)
    plt.imshow(img, 'gray')
    plt.title('original image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(232)
    plt.imshow(fft_show)
    plt.title('FFT')
    plt.xticks([]), plt.yticks([])

    plt.subplot(233)
    plt.imshow(img_recor1, 'gray')
    plt.title('IFFT')
    plt.xticks([]), plt.yticks([])

    plt.subplot(234)
    plt.imshow(img, 'gray')
    plt.title('original image')

    plt.subplot(235)
    plt.imshow(img_dct_log)
    plt.title('DCT(cv2_dct)')

    plt.subplot(236)
    plt.imshow(img_recor2, 'gray')
    plt.title('IDCT(cv2_idct)')

    plt.show()


if __name__ == '__main__':
    # data
    img = cv2.imread('E:/Desktop/easyCS/data/limotiff/6.tif', 0)
    img = img.astype('float')

    # img = cv2.imread("E:/Desktop/easyCS/data/limotiff/lena.jpg")
    # img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.floa
    # t32)

    # para
    mycent_num = 30
    # show
    show_compress_data(img=img, mycent=mycent_num)

    print('[*] 百分比是 {:.2f}'.format(mycent_num / img.shape[0] * 100))
    print('[*] Done')
