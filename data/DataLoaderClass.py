import subprocess
import os
import multiprocessing
import cv2
import numpy as np
from config import args_config
import time

image_data_souce = list()


class ImageData():
    # BASIC
    image_path = []
    images = []
    image_nums = 0
    num_process = 0
    # Image info
    prefix_Image = []
    width, height, depth = 0, 0, 0
    start_num, end_num, = 0, 0

    def __init__(self, image_path, width, height, depth, start_num, end_num, prefix_Image, num_process):
        self.image_path = image_path
        self.width, self.height, self.depth = width, height, depth
        self.start_num, self.end_num = start_num, end_num
        self.prefix_Image = prefix_Image
        self.num_process = num_process
        self.load_images_init()

    def generate_train_test_data_NONE(self, verbose=0, testselect=10):
        train_X = list()
        test_X = list()
        for i in range(len(self.images)):
            x_t = np.array(self.images[i])
            if verbose >= 1:
                print("num {}  ||  mod {} ".format(i, i % testselect))
            if i % testselect > 0:
                train_X.append(x_t)
            else:
                test_X.append(x_t)
        train_X = np.stack(train_X, axis=0)
        test_X = np.stack(test_X, axis=0)
        train_Y = train_X
        test_Y = test_X
        return [train_X, train_Y, test_X, test_Y]

    def load_images_init(self):
        # init
        self.images.clear()
        image_data_souce.clear()
        # load
        load_images_data(self.image_path, self.prefix_Image,
                         self.start_num, self.end_num,
                         self.width, self.height,
                         self.num_process)
        # copy
        self.images = image_data_souce.copy()
        image_data_souce.clear()
        self.image_nums = len(self.images)

    @staticmethod
    def image_full_path(image_path, prefix_Image, i):
        return os.path.join(image_path, prefix_Image + "%05d.tif" % i)

    def print_img_info(self):
        log = '[{} Show Image Info {}]\n'.format("*" * 30, "*" * 30)
        log += 'Image_Path                     : {}\n'.format(self.image_path)
        log += 'Width x Height   Depth         : {:<4} x {:<4} {:<4}\n'.format(self.width, self.height, self.depth)
        log += 'Start - End      Images_nums   : {:<4} - {:<4} {:<4}\n'.format(self.start_num, self.end_num,
                                                                               self.image_nums)
        log += 'prefix_Image                   : {}\n'.format(self.prefix_Image)
        log += 'num_process                    : {}\n'.format(self.num_process)
        log += '[{}]'.format("*" * 78)
        print(log)


def func_time(func):
    def inner(*args, **kw):
        start_time = time.time()
        func(*args, **kw)
        end_time = time.time()
        print('[ {} ]运行时间为：{} s'.format(func, end_time - start_time))

    return inner


@func_time
def load_images_data(image_path, prefix_Image, start_num, end_num, width, height, num_process):
    pool = multiprocessing.Pool(processes=num_process)
    for i in range(start_num, end_num + 1):
        pool.apply_async(
            func=load_single_image,
            args=(image_path, prefix_Image, i, width, height),
            callback=update_image_path,
            error_callback=error_function
        )
    pool.close()
    pool.join()


def load_single_image(image_path, prefix_Image, i, width, height):
    image_path = ImageData.image_full_path(image_path, prefix_Image, i)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img.shape[0] != width or img.shape[1] != height:
        imgResize = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        imgResize = img
    x = np.array(imgResize / 255.)
    print("[ * ] Loading {}".format(image_path))
    return x


def update_image_path(result):
    image = result
    image_data_souce.append(image)


def error_function(error):
    """ error callback called when an encoding job in a worker process encounters an exception
    """
    print('***** ATTENTION %s', type(error))
    print('***** ATTENTION %s', repr(error))


def listdir_full_path(directory):
    """ like os.listdir(), but returns full absolute paths
    """
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            yield os.path.abspath(os.path.join(directory, f))


def split_path(path):
    (filepath, file_full_name) = os.path.split(path)
    filename, extension = os.path.splitext(file_full_name)
    return filepath, file_full_name, filename, extension


if __name__ == '__main__':
    args = args_config()
    image_path = args.data_path
    width = args.img_size_x
    height = args.img_size_y
    depth = args.img_depth
    start_num = args.data_star_num
    end_num = args.data_end_num
    prefix_Image = args.prefix_Image
    num_process = args.num_process

    data = ImageData('./17782/', width, height, depth, start_num, end_num, prefix_Image, num_process)

    data.print_img_info()
    train_X, train_Y, test_X, test_Y = data.generate_train_test_data_NONE(testselect=10)

    print(" {} {} {} {}".format(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape))
    # data = ImageData(path)
    # data.print_basic_info()
