# coding:utf-8
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DCT基作为稀疏基，重建算法为OMP算法 ，图像按列进行处理
# 参考文献: 任晓馨. 压缩感知贪婪匹配追踪类重建算法研究[D].
# 北京交通大学, 2012.
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 导入所需的第三方库文件
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


# OMP算法函数
def cs_omp(y, D):  # 传入参数为y和传感矩阵
    L = math.floor(y.shape[0] / 4)
    residual = y  # 初始化残差
    index = np.zeros((L), dtype=int)
    for i in range(L):
        index[i] = -1
    result = np.zeros(256)
    for j in range(L):  # 迭代次数
        product = np.fabs(np.dot(D.T, residual))
        pos = np.argmax(product)  # 最大投影系数对应的位置
        index[j] = pos
        tmp = []
        for tt in range(len(index)):
            if (index[tt] > 0) or (index[tt] == 0):
                tmp.append(tt)
        # tmp1 = D[:, tmp]
        my = np.linalg.pinv(D[:, tmp])  # 最小二乘
        a = np.dot(my, y)  # 最小二乘
        residual = y - np.dot(D[:, tmp], a)
    result[tmp] = a
    return result


if __name__ == "__main__":
    # 读取图像，并变成numpy类型的 array
    im = cv2.imread("./data/lena.jpg")
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_CUBIC)

    # 生成高斯随机测量矩阵
    sampleRate = 0.5  # 采样率
    Phi = np.random.randn(int(256 * sampleRate), 256)

    # 生成稀疏基DCT矩阵
    mat_dct_1d = np.zeros((256, 256))
    v = range(256)
    for k in range(0, 256):
        dct_1d = np.cos(np.dot(v, k * math.pi / 256))
        if k > 0:
            dct_1d = dct_1d - np.mean(dct_1d)
        mat_dct_1d[:, k] = dct_1d / np.linalg.norm(dct_1d)

    # 随机测量
    img_cs_1d = np.dot(Phi, im)

    # 重建
    sparse_rec_1d = np.zeros((256, 256))  # 初始化稀疏系数矩阵
    Theta_1d = np.dot(Phi, mat_dct_1d)  # 测量矩阵乘上基矩阵
    for i in range(256):
        print('正在重建第', i, '列。')
        column_rec = cs_omp(img_cs_1d[:, i], Theta_1d)  # 利用OMP算法计算稀疏系数
        sparse_rec_1d[:, i] = column_rec
    img_rec = np.dot(mat_dct_1d, sparse_rec_1d).astype('uint8')  # 稀疏系数乘上基矩阵

    # 显示重建后的图片
    plt.imshow(img_rec, cmap='gray')
    plt.show()
