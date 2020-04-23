from data.mask_utils import get_mask
from config import args_config_predict
from config import args_config
from data import get_Blocks, assembleBlocks

import os
import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt


def generate_train_test_data(data_path, start_num, end_num, mask, testselect=10, verbose=0):
    prefix_Image = r"17782_"
    x = list()
    for i in range(start_num, end_num + 1):
        if verbose >= 1:
            print("[@data_loader] {} is loading".format(os.path.join(data_path, prefix_Image + "%05d.tif" % i)))
        width = height = 256
        img = cv2.imread(os.path.join(data_path, prefix_Image + "%05d.tif" % i), cv2.IMREAD_GRAYSCALE)
        imgResize = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        x.append(imgResize)
    train_X = list()
    train_Y = list()
    test_X = list()
    test_Y = list()
    for i in range(len(x)):
        x_t = to_bad_img(x[i] / 255, mask)
        y_t = np.array(x[i] / 255)
        # x_t = np.resize(x_t, (1, x_t.shape[0], x_t.shape[1]))
        # y_t = np.resize(y_t, (1, y_t.shape[0], x_t.shape[1]))
        if verbose >= 1:
            print("num {}  ||  mod {} ".format(i, i % testselect))
        if i % testselect > 0:
            train_X.append(x_t)
            train_Y.append(y_t)
        else:
            test_X.append(x_t)
            test_Y.append(y_t)
    train_X = np.concatenate(train_X, 0)
    train_Y = np.concatenate(train_Y, 0)
    test_X = np.concatenate(test_X, 0)
    test_Y = np.concatenate(test_Y, 0)
    return [train_X, train_Y, test_X, test_Y]


def generate_bigimage(data_path, indx, mask, Subimg_size_x=256, Subimg_size_y=256, overlap_percent=0, verbose=0):
    prefix_Image = r"17782_"
    if verbose >= 1:
        print("[@data_loader] {} is loading ...".format(os.path.join(data_path, prefix_Image + "%05d.tif" % indx)))
    img = cv2.imread(os.path.join(data_path, prefix_Image + "%05d.tif" % indx), cv2.IMREAD_GRAYSCALE)
    blocks = get_Blocks(img, Subimg_size_x, Subimg_size_y, overlap_percent, verbose=verbose)
    X = list()
    Y = list()
    for i in range(len(blocks)):
        x_t = to_bad_img(blocks[i, :, :] / 255, mask)
        y_t = np.array(blocks[i, :, :] / 255)
        x_t = np.resize(x_t, (1, x_t.shape[0], x_t.shape[1]))
        y_t = np.resize(y_t, (1, y_t.shape[0], x_t.shape[1]))
        X.append(x_t)
        Y.append(y_t)
    X = np.concatenate(X, 0)
    Y = np.concatenate(Y, 0)
    return img.shape[0], img.shape[1], X, Y


def to_bad_img(x, mask):
    gray = (x + 1.) / 2.
    fft = scipy.fftpack.fft2(gray)
    fft = scipy.fftpack.fftshift(fft)
    par_fft = fft * mask
    ifft = scipy.fftpack.ifftshift(par_fft)
    x = scipy.fftpack.ifft2(ifft)
    x = np.abs(x)
    x = x * 2 - 1
    return x


def get_test_image(x_data, y_data, num=20):
    indx = np.arange(x_data.shape[0])
    np.random.shuffle(indx)
    x_test = x_data[indx[0:num], :, :]
    y_test = y_data[indx[0:num], :, :]
    return x_test, y_test


if __name__ == '__main__':
    # mask_name = "poisson2d"
    # mask_perc = 1
    # print('[*] run basic configs ... ')
    # args = args_config_predict()
    # print('[*] loading mask ... ')
    # mask = get_mask(mask_name=args.maskname, mask_perc=args.maskperc, mask_path="mask")
    # print('[*] load data ... ')
    # [x, y] = generate_bigimage("../data/17782/", indx=20, mask=mask, verbose=1)
    # =================================== BASIC CONFIGS =================================== #
    print('[*] run basic configs ... ')
    args = args_config()
    # ==================================== PREPARE DATA ==================================== #
    print('[*] loading mask ... ')
    mask = get_mask(mask_name=args.maskname, mask_perc=args.maskperc, mask_path="E:/Desktop/easyCS/data/mask/")
    print('[*] load data ... ')
    [x, y, a, b] = generate_train_test_data("E:/Desktop/easyCS/data/17782/", 1, 20, mask, verbose=0)
