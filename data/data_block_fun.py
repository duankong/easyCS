import numpy as np


def get_Blocks(data, Subimg_size_x, Subimg_size_y, overlap_percent, verbose=0):
    if len(data.shape) == 2:
        Image_size_x, Image_size_y = data.shape[0], data.shape[1]
    else:
        Image_size_x, Image_size_y = data.shape[1], data.shape[2]
    if verbose > 0:
        print('\033[1;34m')
        print('Input Image size : {:05d} * {:05d}'.format(Image_size_x, Image_size_y))
    if overlap_percent >= 1:
        overlap = int(overlap_percent)
    else:
        overlap = int(np.ceil(Subimg_size_y * overlap_percent))  # overlap pixel
    if overlap * 2 >= Subimg_size_x or overlap * 2 >= Subimg_size_y:
        print('\033[1;31m')
        print('get_Blocks :: ===> Error 2*overlap should be small than x and y !')
        exit()
    n_vert = int(np.ceil(Image_size_x / Subimg_size_x))  # vert
    n_horiz = int(np.ceil(Image_size_y / Subimg_size_y))  # horiz
    if verbose >= 2:
        print('\033[0;m')
        print('overlap : ', overlap, 'n_vert : ', n_vert, 'n_horiz : ', n_horiz)
    #  Pad image, check new shape
    if ~(n_vert * Subimg_size_x == Image_size_x):
        pad_ver = n_vert * Subimg_size_x - Image_size_x
    else:
        pad_ver = 0
    if ~(n_horiz * Subimg_size_y == Image_size_y):
        pad_hor = n_horiz * Subimg_size_y - Image_size_y
    else:
        pad_hor = 0
    big_img = np.pad(data, ((0, int(pad_ver)), (0, int(pad_hor))), 'constant')  # pad all img
    blocks = np.zeros([int(n_vert * n_horiz), int(Subimg_size_x + 2 * overlap), int(Subimg_size_y + 2 * overlap)])
    if verbose > 0:
        print('\033[1;34m')
        print('Blocks size  :', blocks.shape)
    padded_img = np.pad(big_img, ((overlap, overlap), (overlap, overlap)), 'edge')  # pad overlap
    # print(blocks.shape)
    # Iterate through the image and append to 'blocks.'
    for i in range(n_vert):
        for j in range(n_horiz):
            x_indx = [(i * Subimg_size_x), ((i + 1) * Subimg_size_x + 2 * overlap)]
            y_indx = [j * Subimg_size_y, ((j + 1) * Subimg_size_y + 2 * overlap)]
            # print(x_indx, y_indx)
            blocks[int(n_horiz * i + j), :, :] = padded_img[x_indx[0]:x_indx[1], y_indx[0]:y_indx[1]]
    print('\033[0m')
    return blocks


def assembleBlocks(blocks, Image_size_x=512, Image_size_y=512, Subimg_size_x=256, Subimg_size_y=256, overlap_percent=0):
    if overlap_percent >= 1:
        overlap = int(overlap_percent)
    else:
        overlap = int(np.ceil(Subimg_size_y * overlap_percent))  # overlap pixel
    n_vert = int(np.ceil(Image_size_x / Subimg_size_x))  # n_vert
    n_horiz = int(np.ceil(Image_size_y / Subimg_size_y))  # n_horiz
    new_image = np.zeros([int(Image_size_x + 2 * overlap), int(Image_size_y + 2 * overlap)])
    #  Pad image, check new shape
    if ~(n_vert * Subimg_size_x == Image_size_x):
        pad_ver = n_vert * Subimg_size_x - Image_size_x
    else:
        pad_ver = 0
    if ~(n_horiz * Subimg_size_y == Image_size_y):
        pad_hor = n_horiz * Subimg_size_y - Image_size_y
    else:
        pad_hor = 0
    big_img = np.pad(new_image, ((0, int(pad_ver)), (0, int(pad_hor))), 'edge')  # pad all img
    #  block mask for alpha blending
    x_shape = blocks.shape[1]
    y_shape = blocks.shape[2]
    block_mask_hang = np.ones([x_shape, y_shape])
    for i_indx in range(overlap * 2):
        block_mask_hang[i_indx, :] = block_mask_hang[i_indx, :] * (1.0 / (2 * overlap + 1)) * (i_indx + 1)
        block_mask_hang[x_shape - (i_indx + 1), :] = block_mask_hang[i_indx, :]
    block_mask_lie = np.ones([x_shape, y_shape])
    for i_indx in range(overlap * 2):
        block_mask_lie[:, i_indx] = block_mask_lie[:, i_indx] * (1.0 / (2 * overlap + 1)) * (i_indx + 1)
        block_mask_lie[:, y_shape - (i_indx + 1)] = block_mask_lie[:, i_indx]
    # i~=0,j~=0 cornel
    block_mask1 = np.ones([x_shape, y_shape])
    for i_indx in range(0, overlap * 2 + 1):
        block_mask1[:, i_indx] = block_mask1[:, i_indx] * (1.0 / (2 * overlap + 1)) * (i_indx + 1)
        block_mask1[:, y_shape - (i_indx + 1)] = block_mask1[:, i_indx]
        block_mask1[i_indx, :] = block_mask1[i_indx, :] * (1.0 / (2 * overlap + 1)) * (i_indx + 1)
        block_mask1[x_shape - (i_indx + 1), :] = block_mask1[i_indx, :]
    # i==0,j~=0 列，上方元素改变
    block_mask2 = np.concatenate([block_mask_lie[0:2 * overlap, :], block_mask1[2 * overlap:x_shape + 1, :]])
    # i==end,j~=0 ,j~=end 下方元素未改变
    block_mask6 = np.concatenate([block_mask1[0:2 * overlap, :], block_mask_lie[2 * overlap:x_shape + 1, :]])
    # i~=0,j==0 行，左方元素改变
    block_mask3 = np.concatenate([block_mask_hang[:, 0:2 * overlap], block_mask1[:, 2 * overlap:y_shape]], axis=1)
    # i~=0,i~=end,j=end 右方元素未改变
    block_mask5 = np.concatenate([block_mask1[:, 0:2 * overlap], block_mask_hang[:, 2 * overlap:y_shape]], axis=1)
    # i==0,j==0 行，左方和上方元素改变
    block_mask4 = np.copy(block_mask1)
    block_mask4[0:(x_shape - 2 * overlap), 0:(y_shape - 2 * overlap)] = 1
    block_mask4[(x_shape - 2 * overlap):x_shape, 0:(y_shape - 2 * overlap)] = block_mask_hang[
                                                                              (x_shape - 2 * overlap):x_shape,
                                                                              0:(y_shape - 2 * overlap)]
    block_mask4[0:(x_shape - 2 * overlap), (y_shape - 2 * overlap):y_shape] = block_mask_lie[
                                                                              0:(x_shape - 2 * overlap),
                                                                              (y_shape - 2 * overlap):y_shape]
    # i==0,j=end 右上方
    block_mask7 = np.copy(block_mask1)
    block_mask7[0:(x_shape - 2 * overlap), (y_shape - 2 * overlap):y_shape] = 1
    block_mask7[(x_shape - 2 * overlap):x_shape, (y_shape - 2 * overlap):y_shape] = block_mask_hang[
                                                                                    (x_shape - 2 * overlap):x_shape,
                                                                                    (y_shape - 2 * overlap):y_shape]
    block_mask7[0:(x_shape - 2 * overlap), 0:(y_shape - 2 * overlap)] = block_mask_lie[0:(x_shape - 2 * overlap),
                                                                        0:(y_shape - 2 * overlap)]
    # i==end,j==0 左下方
    block_mask8 = block_mask4[::-1, :]
    # i==end,j==end 右下方
    block_mask9 = block_mask7[::-1, :]
    # Iterate through the image and append to 'blocks.'
    for i_indx in range(0, n_vert):
        for j in range(0, n_horiz):
            # print(i_indx,j)
            # Alpha Blending - multiply each block by block mask and add to image
            x_indx = [i_indx * Subimg_size_x, ((i_indx + 1) * Subimg_size_x + 2 * overlap)]
            y_indx = [j * Subimg_size_y, ((j + 1) * Subimg_size_y + 2 * overlap)]
            # print(x_indx, y_indx)
            if i_indx == 0 and j == 0:
                block_mask = block_mask4  # left up
            elif i_indx == 0 and j == n_horiz - 1:
                block_mask = block_mask7  # right up
            elif i_indx == n_vert - 1 and j == 0:
                block_mask = block_mask8  # left down
            elif i_indx == n_vert - 1 and j == n_horiz - 1:
                block_mask = block_mask9  # right down
            elif i_indx == 0 and j != 0 and j != n_horiz - 1:
                block_mask = block_mask2  # up
            elif i_indx != 0 and j == 0 and i_indx != n_vert - 1:
                block_mask = block_mask3  # left
            elif i_indx == n_vert - 1 and j != 0 and j != n_vert - 1:
                block_mask = block_mask6  # down
            elif i_indx != 0 and i_indx != n_vert - 1 and j == n_horiz - 1:
                block_mask = block_mask5  # right
            else:
                block_mask = block_mask1
            blocks_temp = blocks[n_horiz * i_indx + j, :, :]
            blocks_temp = np.squeeze(blocks_temp)
            temp_num = np.multiply(blocks_temp, block_mask)
            big_img[x_indx[0]: x_indx[1], y_indx[0]: y_indx[1]] = temp_num \
                                                                  + big_img[x_indx[0]:x_indx[1], y_indx[0]:y_indx[1]]
    fix_img = big_img[overlap:(overlap + Image_size_x), overlap:(overlap + Image_size_y)]
    return fix_img
