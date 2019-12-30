import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt


def get_mask(mask_name, mask_perc, mask_path="mask",verbose=0):
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


if __name__ == '__main__':
    mask_name = "gaussian2d", "gaussian1d", "poisson2d"
    mask_perc = 1, 5, 10, 20, 30, 40, 50
    mask = list()
    for i, mask_perc in enumerate(mask_perc):
        mask.append(get_mask(mask_name[2], mask_perc, mask_path="mask",verbose=1))
