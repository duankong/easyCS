from model import unet
from data import get_mask, generate_bigimage, assembleBlocks
from config import args_config_predict

import torch
from torchsummary import summary
import torch.utils.data as Data

import numpy as np
import os
import cv2


def predict_big_img_sequence():
    # =================================== BASIC CONFIGS =================================== #
    print('[*] run basic configs ... ')
    args = args_config_predict()
    # ==================================== PREPARE DATA ==================================== #
    print('[*] loading mask ... ')
    mask = get_mask(mask_name=args.maskname, mask_perc=args.maskperc, mask_path="data/mask")
    # ==================================== DEFINE MODEL ==================================== #
    print('[*] define model ... ')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_net = unet.UNet(n_channels=args.img_n_channels, n_classes=args.img_n_classes).to(device)
    print('[*] Try resume from checkpoint')
    if os.path.isdir('checkpoint'):
        try:
            checkpoint = torch.load('./checkpoint/' + args.model_name, map_location=device)
            print('===> Load last checkpoint data')
            my_net.load_state_dict(checkpoint['state'])  # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            print("===> loaded checkpoint '{}' (trained for {} epochs,the best loss is {:.6f})".format(args.model_name,
                                                                                                       checkpoint[
                                                                                                           'epoch'],
                                                                                                       best_loss))
        except FileNotFoundError:
            print('Can\'t found ' + args.model_name)
            return 0
    if args.model_show:
        summary(my_net, input_size=(2, args.img_size_x, args.img_size_y))
    # ==================================== PREDICT MODEL ==================================== #
    print('[*] start predict ...')
    for indx in range(args.test_star_num, args.test_end_num + 1):
        print('[@] load data ... ')
        [Image_size_x,Image_size_y,x, y] = generate_bigimage(args.test_path, indx, mask, Subimg_size_x=args.Subimg_size_x,
                                   Subimg_size_y=args.Subimg_size_y,
                                   overlap_percent=args.overlap_percent,
                                   verbose=0)
        x_data = torch.from_numpy(x[:]).float().unsqueeze(1)
        y_data = torch.from_numpy(y[:]).float().unsqueeze(1)
        if torch.cuda.is_available():
            x_data, y_data = x_data.cuda(), y_data.cuda()
        # print("x_data shape is [{}],y_data shape is [{}]".format(x_data.shape, y_data.shape))
        predict_data = Data.DataLoader(dataset=Data.TensorDataset(x_data, y_data), batch_size=args.test_batch_size,
                                       shuffle=False, num_workers=4)
        test_predict = list()
        test_real = []
        for step, (b_x, b_y) in enumerate(predict_data):  # gives batch data, normalize x when iterate train_loader
            with torch.no_grad():
                test_predict.append(my_net(b_x).cpu().data.numpy())
                test_real.append(b_y.cpu().data.numpy())
        test_predict = np.concatenate(test_predict, axis=0).squeeze()
        test_real = np.concatenate(test_real, axis=0).squeeze()
        if len(test_predict.shape) == 3:
            y_real = assembleBlocks(test_real, Image_size_x=Image_size_x, Image_size_y=Image_size_y, Subimg_size_x=args.Subimg_size_x,
                                    Subimg_size_y=args.Subimg_size_y, overlap_percent=args.overlap_percent)
            x_predict = assembleBlocks(test_predict, Image_size_x=Image_size_x, Image_size_y=Image_size_y,
                                       Subimg_size_x=args.Subimg_size_x,
                                       Subimg_size_y=args.Subimg_size_y, overlap_percent=args.overlap_percent)
        else:
            y_real = test_real
            x_predict = test_predict
        x_predict = np.uint8(x_predict * 255)
        x_predict_name = args.test_target_path + 'test_17782_' + "%05d_predict.tif" % indx
        cv2.imwrite(x_predict_name, x_predict)
        if args.save_real:
            y_real_name = args.test_target_path + 'test_17782_' + "%05d_real.tif" % indx
            y_real = np.uint8(np.squeeze(y_real) * 255)
            cv2.imwrite(y_real_name, y_real)
    # ==================================== PREDICT DONE ==================================== #
    print("[*] predict Done!")


if __name__ == '__main__':
    predict_big_img_sequence()
