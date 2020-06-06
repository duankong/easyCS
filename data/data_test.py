import numpy as np
from data import assembleBlocks, get_Blocks
import os
from skimage import io
import cv2

if __name__ == '__main__':
    img_2 = cv2.imread('./lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow("Image")
    cv2.imshow("Image", img_2)
    data = np.squeeze(img_2)
    Subimg_size_x = 32
    Subimg_size_y = 65
    overlap_percent = 7
    Image_size_x, Image_size_y = 256, 256
    blocks = get_Blocks(data, Subimg_size_x, Subimg_size_y, overlap_percent, verbose=1)
    fix_img = assembleBlocks(blocks, Image_size_x, Image_size_y, Subimg_size_x, Subimg_size_y, overlap_percent)
    fix_img = np.round(fix_img)
    error_image = data - fix_img
    cv2.imshow("new_image2", np.uint8(fix_img))
    error_image = np.uint8(error_image)
    cv2.waitKey(2000)
    error = sum(abs(error_image.ravel()))
    print(error)
