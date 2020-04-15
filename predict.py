from model import unet
from data import generate_train_test_data, get_mask
from config import args_config_predict

import torch
from torchsummary import summary
import torch.utils.data as Data

import numpy as np
import os
import cv2


def predict_img_sequence():
    # =================================== BASIC CONFIGS =================================== #
    print('[*] run basic configs ... ')
    args = args_config_predict()
    # ==================================== PREPARE DATA ==================================== #
    print('[*] loading mask ... ')
    mask = get_mask(mask_name=args.maskname, mask_perc=args.maskperc, mask_path="data/mask")
    print('[*] load data ... ')
    [x, y] = generate_traindata(args.test_path, args.test_star_num, args.test_end_num, mask=mask, verbose=0)
    x_data = torch.from_numpy(x[:]).float().unsqueeze(1)
    y_data = torch.from_numpy(y[:]).float().unsqueeze(1)
    if torch.cuda.is_available():
        x_data, y_data = x_data.cuda(), y_data.cuda()
        print('====> Running on GPU <===')
    print("x_data shape is [{}],y_data shape is [{}]".format(x_data.shape, y_data.shape))
    predict_data = Data.DataLoader(dataset=Data.TensorDataset(x_data, y_data), batch_size=args.test_batch_size,
                                   shuffle=False, num_workers=4)
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
                                                                                                       start_epoch,
                                                                                                       best_loss))
        except FileNotFoundError:
            print('Can\'t found ' + args.model_name)
            return 0
    if args.model_show:
        summary(my_net, input_size=(2, args.img_size_x, args.img_size_y))
    print('[*] start predict ...')
    for step, (b_x, b_y) in enumerate(predict_data):  # gives batch data, normalize x when iterate train_loader
        with torch.no_grad():
            test_output = my_net(b_x)

        test_output = test_output.cpu().data.numpy()
        test_output = np.squeeze(test_output)
        if len(test_output.shape) == 2:
            x_predict = np.uint8(test_output * 255)
            pic_num = args.test_star_num + step
            x_predict_name = args.test_target_path + 'test_17782_' + "%05d_predict.tif" % pic_num
            cv2.imwrite(x_predict_name, x_predict)
            if args.save_real:
                y_real_name = args.test_target_path + 'test_17782_' + "%05d_real.tif" % pic_num
                y_real = np.uint8(np.squeeze(b_y) * 255)
                cv2.imwrite(y_real_name, y_real)
        else:
            for k in range(test_output.shape[0]):
                x_predict = np.uint8(np.squeeze(test_output[k, :, :]) * 255)
                pic_num = args.test_star_num + step * test_output.shape[0] + k
                x_predict_name = args.test_target_path + 'test_17782_' + "%05d_predict.tif" % pic_num
                print("[...] {}".format(x_predict_name))
                cv2.imwrite(x_predict_name, x_predict)
                if args.save_real:
                    y_real = np.uint8(np.squeeze(b_y[k, :, :]) * 255)
                    y_real_name = args.test_target_path + 'test_17782_' + "%05d_real.tif" % pic_num
                    cv2.imwrite(y_real_name, y_real)
    print("[*] predict Done!")


if __name__ == '__main__':
    predict_img_sequence()
