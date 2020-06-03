import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy, cv2, pywt


def get_mask(mask_name, mask_perc, mask_path="mask", verbose=0):
    if verbose >= 1:
        print('[*] loading mask ... ')
    if mask_name == "gaussian2d":
        mask = \
            loadmat(
                os.path.join(os.path.join(mask_path, 'Gaussian2D'),
                             "GaussianDistribution2DMask_{}.mat".format(mask_perc)))[
                'maskRS2']
    elif mask_name == "gaussian1d":
        mask = \
            loadmat(
                os.path.join(os.path.join(mask_path, 'Gaussian1D'),
                             "GaussianDistribution1DMask_{}.mat".format(mask_perc)))[
                'maskRS1']
    elif mask_name == "poisson2d":
        mask = \
            loadmat(
                os.path.join(os.path.join(mask_path, 'Poisson2D'),
                             "PoissonDistributionMask_{}.mat".format(mask_perc)))[
                'population_matrix']
    else:
        raise ValueError("no such mask exists: {}".format(mask_name))
    mask = np.array(mask)
    if verbose >= 1:
        totolnum = np.sum(mask == 1)
        log = "==> name={} perc={:02d} percent={:.2f} shape={}*{}".format(mask_name, mask_perc, totolnum / 256 / 256,
                                                                          mask.shape[0], mask.shape[1])
        print(log)
        plt.clf()
        plt.imshow(mask, cmap='gray')
        plt.show()

    return mask


def show_fft_mask(x, mask):
    orgin = x
    gray = (orgin + 1.) / 2.

    fft = scipy.fftpack.fft2(gray)
    fft = scipy.fftpack.fftshift(fft)

    par_fft = fft * mask

    ifft = scipy.fftpack.ifftshift(par_fft)
    x = scipy.fftpack.ifft2(ifft)
    x = np.abs(x)
    x = x * 2 - 1

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(orgin, cmap='gray')
    plt.title('origian')

    plt.subplot(2, 3, 2)
    plt.title("part recon x")
    plt.imshow(x, cmap='gray')

    ift_data = scipy.fftpack.ifftshift(fft)
    ift_img = abs(scipy.fftpack.ifft2(ift_data))
    plt.subplot(2, 3, 3)
    plt.title('full recon')
    plt.imshow(ift_img, cmap='gray')

    ft_img = np.log(abs(fft) + 1)
    plt.subplot(2, 3, 4)
    plt.title('full fft')
    plt.imshow(ft_img, cmap='gray')

    par_ft_img = np.log(abs(par_fft) + 1)
    plt.subplot(2, 3, 5)
    plt.title('part fft')
    plt.imshow(par_ft_img, cmap='gray')

    par_ft_img = mask
    plt.subplot(2, 3, 6)
    plt.title('mask ')
    plt.imshow(par_ft_img, cmap='gray')

    plt.show()


def show_dct_mask(img, mask):
    # img = pywt.data.aero().astype(np.float)
    # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img_dct = cv2.dct(img.astype(np.float))  # 进行离散余弦变换
    img_dct_log = np.log(abs(img_dct))  # 进行log处理
    img_recon = cv2.idct(img_dct)  # 进行离散余弦反变换

    plt.figure()
    plt.subplot(2, 4, 1)
    plt.title('Original img')
    plt.imshow(img, cmap='gray')

    plt.subplot(2, 4, 2)
    plt.title('full dct')
    plt.imshow(img_dct, cmap='gray')

    plt.subplot(2, 4, 3)
    plt.title('full dct log')
    plt.imshow(img_dct_log, cmap='gray')

    plt.subplot(2, 4, 4)
    plt.title('recon')
    plt.imshow(img_recon, cmap='gray')

    plt.subplot(2, 4, 6)
    plt.title('part dct')
    plt.imshow(img_dct * mask, cmap='gray')

    plt.subplot(2, 4, 7)
    plt.title('part dct log')
    plt.imshow((img_dct_log * mask), cmap='gray')

    plt.subplot(2, 4, 8)
    plt.title('part dct recon')
    plt.imshow(cv2.idct(img_dct * mask), cmap='gray')

    plt.show()


if __name__ == '__main__':
    mask_name = "gaussian2d", "gaussian1d", "poisson2d"
    mask_perc = [1, 5, 10, 20, 30, 40, 50]
    mask = list()
    for i, mask_indx in enumerate(mask_perc):
        mask.append(get_mask(mask_name[0], mask_indx, mask_path="mask", verbose=0))
    # data
    img = cv2.imread(os.path.join("17782", r"17782_" + "%05d.tif" % 1), cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("lena.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    imgResize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    # show
    show_dct_mask(imgResize - 128, mask[6])
    show_fft_mask(imgResize, mask[0])
    # show mask detail
    for i, mask_temp in enumerate(mask):
        sum = np.sum(mask_temp) / 256 / 256 * 100
        print('[#] mask {} is {}%'.format(mask_perc[i], sum))
