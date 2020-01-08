from model import unet, FeatureExtractor
from data import generate_traindata, get_mask, get_test_image
from config import args_config
from utils import get_para_log, get_model, ssim

import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
from tensorboardX import SummaryWriter
import torchvision

import numpy as np
import os, time
from torchsummary import summary
import skimage.metrics


def main_train():
    # =================================== BASIC CONFIGS =================================== #
    print('[*] run basic configs ... ')
    args = args_config()
    writer = SummaryWriter(os.path.join("runs/", args.model_log))
    # write para into a text
    para_data = os.path.join("runs/", args.model_log) + "/para_data.txt"
    with open(para_data, "w") as file:  # ”w"代表着每次运行都覆盖内容
        file.write(get_para_log(args))
    # ==================================== PREPARE DATA ==================================== #
    print('[*] loading mask ... ')
    mask = get_mask(mask_name=args.maskname, mask_perc=args.maskperc, mask_path="data/mask")
    print('[*] load data ... ')
    [x, y] = generate_traindata(args.data_path, args.data_star_num, args.data_end_num, mask, verbose=0)
    if args.model == "Unet_conv":
        x = y
    x_data = torch.from_numpy(x[:])
    y_data = torch.from_numpy(y[:])
    x_data, y_data = x_data.float(), y_data.float()
    x_data = x_data.unsqueeze(1)
    y_data = y_data.unsqueeze(1)
    if torch.cuda.is_available():
        x_data, y_data = x_data.cuda(), y_data.cuda()
        print('====> Running on GPU <===')
    print("x_data shape is [{}],y_data shape is [{}]".format(x_data.shape, y_data.shape))
    deal_dataset = Data.TensorDataset(x_data, y_data)
    train_loader = Data.DataLoader(dataset=deal_dataset, batch_size=args.batch_size, shuffle=True)
    x_test, y_test = get_test_image(x_data, y_data, 20)
    img_grid_y = torchvision.utils.make_grid(y_test, nrow=5)
    img_grid_x = torchvision.utils.make_grid(x_test, nrow=5)
    writer.add_image('img_test/ground', img_grid_y)
    writer.add_image('img_test/input', img_grid_x)
    # ==================================== DEFINE MODEL ==================================== #
    print('[*] define model ... ')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_net_G = get_model(model_name=args.model, n_channels=args.img_n_channels, n_classes=args.img_n_classes)
    demo_input = torch.rand(1, 1, 256, 256)
    writer.add_graph(my_net_G, input_to_model=demo_input)
    my_net_G.to(device)
    print('[*] Try resume from checkpoint')
    if os.path.isdir('checkpoint'):
        try:
            checkpoint = torch.load('./checkpoint/' + args.model_name)
            print('==> Load last checkpoint data')

            my_net_G.load_state_dict(checkpoint['state'])  # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            print("==> Loaded checkpoint '{}' (trained for {} epochs,the best loss is {:.6f})".format(args.model_name,
                                                                                                      checkpoint[
                                                                                                          'epoch'],
                                                                                                      best_loss))
        except FileNotFoundError:
            start_epoch = 0
            best_loss = 10
            print('==> Can\'t found ' + args.model_name)
    else:
        start_epoch = 0
        best_loss = np.inf
        print('==> Start from scratch')
    if args.model_show:
        summary(my_net_G, input_size=(1, args.img_size_x, args.img_size_y))
    # ==================================== DEFINE TRAIN OPTS ==================================== #
    print('[*] define training options ... ')
    optimizer_G = optim.Adam(my_net_G.parameters(), lr=args.lr)  # optimize all net parameters
    # ==================================== DEFINE LOSS ==================================== #
    print('[*] define loss functions ... ')
    loss_mse = nn.MSELoss()
    vgg_Feature_model = FeatureExtractor().to(device)
    # ==================================== TRAINING ==================================== #
    print('[*] start training ... ')
    for epoch in range(start_epoch, args.epochs):
        for step, (train_x, train_y) in enumerate(
                train_loader):  # gives batch data, normalize x when iterate train_loader
            iter_num = (epoch) * len(train_loader) * args.batch_size + step * args.batch_size
            optimizer_G.zero_grad()  # clear gradients for this training step
            # Generate a batch of images
            g_img = my_net_G(train_x)  # get output
            # Loss measures generator's ability to fool the discriminator
            loss_g_mse = loss_mse(g_img, train_y)
            if args.loss_mse_only == True:
                g_loss = loss_g_mse
            else:
                g_loss = args.alpha * loss_g_mse
                if args.loss_ssim == True:
                    loss_g_ssim = 1 - ssim(g_img, train_y)
                    g_loss += args.gamma * loss_g_ssim
                if args.loss_vgg == True:
                    loss_g_vgg = loss_mse(vgg_Feature_model(g_img), vgg_Feature_model(train_y))
                    g_loss += args.beta * loss_g_vgg
            g_loss.backward()  # backpropagation, compute gradients
            optimizer_G.step()  # apply gradients
            if step % 2 == 0:
                with torch.no_grad():
                    test_output = my_net_G(x_test)
                psnr_num = skimage.metrics.peak_signal_noise_ratio(
                    y_test.cpu().data.numpy(), test_output.cpu().data.numpy())
                mse_num = skimage.metrics.mean_squared_error(
                    y_test.cpu().data.numpy() * 255, test_output.cpu().data.numpy() * 255)
                log = "[**] Epoch [{:02d}/{:02d}] Step [{:04d}/{:04d}]".format(epoch + 1, args.epochs,
                                                                               (step + 1) * args.batch_size,
                                                                               len(train_loader) * args.batch_size)
                # TensorboardX log and print in command line
                writer.add_scalar("train_loss", g_loss.cpu().data.numpy(), iter_num)
                log += " || TRAIN [loss: {:.6f}]".format(g_loss.cpu().data.numpy())
                writer.add_scalar("train/MSE_loss", loss_g_mse.cpu().data.numpy(), iter_num)
                log += " [MSE: {:.6f}]".format(loss_g_mse.cpu().data.numpy())
                if args.loss_mse_only == False and args.loss_ssim == True:
                    writer.add_scalar("train/SSIM_loss", loss_g_ssim.cpu().data.numpy(), iter_num)
                    log += "  [SSIM: {:.4f}]".format(loss_g_ssim.cpu().data.numpy())
                if args.loss_mse_only == False and args.loss_vgg == True:
                    writer.add_scalar("train/VGG_loss", loss_g_vgg.cpu().data.numpy(), iter_num)
                    log += "  [VGG: {:.4f}]".format(loss_g_vgg.cpu().data.numpy())
                writer.add_scalar("test/test_MSE", mse_num, iter_num)
                log += " || TEST [MSE: {:.4f}]".format(mse_num)
                writer.add_scalar("test/test_PSNR", psnr_num, iter_num)
                log += " [PSNR: {:.4f}]".format(psnr_num)
                print(log)

            if g_loss.cpu().data.numpy() < best_loss and step % 20 == 0 and args.model_save:
                # 保存模型示例代码
                best_loss = g_loss.cpu().data.numpy()
                state = {
                    'state': my_net_G.state_dict(),
                    'epoch': epoch,  # 将epoch一并保存
                    'best_loss': g_loss.cpu().data.numpy()
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/' + args.model_name)
                print("[*] Save checkpoints SUCCESS! || loss= {:.5f} epoch= {:03d}".format(best_loss, epoch + 1))
        img_grid = torchvision.utils.make_grid(test_output, nrow=5)
        writer.add_image('img_epoch', img_grid, global_step=epoch)
    writer.close()
    with open(para_data, "a") as file:
        log = "[*] Time is " + time.asctime(time.localtime(time.time())) + "\n"
        log += "=" * 40 + "\n"
        file.write(log)
    print("[*] train Done !")


if __name__ == '__main__':
    main_train()
