import skimage.measure
import numpy as np


def mse_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测
    返回:
    mse -- MSE 评价指标
    """
    y_true = (y_true).ravel() * 255
    y_pred = (y_pred).ravel() * 255
    n = len(y_true)
    mse = sum(np.square(y_true - y_pred)) / n
    return mse


def ssim(x_good, x_bad):
    x_good = np.squeeze(x_good)
    x_bad = np.squeeze(x_bad)
    ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
    return ssim_res


def psnr(x_good, x_bad):
    x_good = np.array(x_good)
    x_good = x_good.squeeze()

    x_bad = np.array(x_bad)
    x_bad = x_bad.squeeze()

    if x_good.ndim == 2:
        PSNR_res = skimage.measure.compare_psnr(x_good, x_bad, 1)

    if x_good.ndim == 3:
        psnr_all = 0
        for indx in range(x_good.shape[0]):
            img1 = x_good[indx, :, :]
            img2 = x_bad[indx, :, :]
            psnr_all += skimage.measure.compare_psnr(img1, img2, 1)
        PSNR_res = psnr_all / x_good.shape[0]
    return PSNR_res


if __name__ == "__main__":
    pass
