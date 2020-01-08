import argparse


def args_config():
    parser = argparse.ArgumentParser(description='train demo')
    # model
    parser.add_argument('--model', type=str, default='Unet_conv',
                        help='the model name：Unet nestedUnet Unet_res Unet_conv')  # [**]
    parser.add_argument('--model_name', type=str, default='Unet_conv_test.t7', help='the model name')  # [**]
    parser.add_argument('--model_log', type=str, default='Unet_conv_test', help='the model log file for check')  # [**]
    parser.add_argument('--model_save', type=bool, default=False, help='the model is save or not')
    parser.add_argument('--model_show', type=bool, default=False, help='the model is show  or not')
    parser.add_argument('--img_n_channels', type=int, default=1, help='the input data channels')
    parser.add_argument('--img_n_classes', type=int, default=1, help='the ouput data classes')
    # train data
    parser.add_argument('--data_path', type=str, default='./data/17782/', help='data path')
    parser.add_argument('--data_star_num', type=int, default=1, help='the first pic num')  # [**]
    parser.add_argument('--data_end_num', type=int, default=201, help='the end pic num')  # [**]
    parser.add_argument('--img_size_x', type=int, default=256, help='the input data size x(row)')
    parser.add_argument('--img_size_y', type=int, default=256, help='the input data size y(col)')
    # mask
    parser.add_argument('--maskname', type=str, default='poisson2d', help='gaussian1d, gaussian2d, poisson2d')  # [**]
    parser.add_argument('--maskperc', type=int, default=10, help='1,5,10,20,30,40,50')  # [**]
    # loss
    parser.add_argument('--loss_mse_only', type=bool, default=False, help='Only use Mse as the loss')
    parser.add_argument('--loss_ssim', type=bool, default=True, help='add SSIM into the loss')
    parser.add_argument('--loss_vgg', type=bool, default=True, help='add vggNet into the loss')
    parser.add_argument('--alpha', type=float, default=15, help='alpha = 15')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma = 0.1')
    parser.add_argument('--beta', type=float, default=0.0025, help='beta = 0.0025')
    # train config
    parser.add_argument('--epochs', type=int, default=100, help='train epoch ')  # [**]
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')  # [**]
    parser.add_argument('--batch_size', type=int, default=8, help='train batch size ')
    return parser.parse_args()


def args_config_predict():
    parser = argparse.ArgumentParser(description='predict demo')
    # model
    parser.add_argument('--model', type=str, default='UNet_res',
                        help='the model name：Unet nestedUnet UNet_res Unet_conv')  # [**]
    parser.add_argument('--model_name', type=str, default='test_log.t7', help='the model name')  # [**]
    parser.add_argument('--model_show', type=bool, default=False, help='the model is show  or not')
    parser.add_argument('--img_n_channels', type=int, default=1, help='the input data channels')
    parser.add_argument('--img_n_classes', type=int, default=1, help='the ouput data classes')
    # test data
    parser.add_argument('--test_path', type=str, default='./data/17782/', help='data path')
    parser.add_argument('--test_target_path', type=str, default='./data/predict/', help='data save path')
    parser.add_argument('--test_star_num', type=int, default=1, help='the first pic num')  # [**]
    parser.add_argument('--test_end_num', type=int, default=10, help='the end pic num')  # [**]
    parser.add_argument('--img_size_x', type=int, default=256, help='the input data size x(row)')
    parser.add_argument('--img_size_y', type=int, default=256, help='the input data size y(col)')
    # test big image
    parser.add_argument('--overlap_percent', type=float, default=0, help='the overlap_percent 0-1 or int>1')
    parser.add_argument('--Subimg_size_x', type=int, default=256, help='the Subimg_size_x')
    parser.add_argument('--Subimg_size_y', type=int, default=256, help='the Subimg_size_y')
    # mask
    parser.add_argument('--maskname', type=str, default='poisson2d', help='gaussian1d, gaussian2d, poisson2d')  # [**]
    parser.add_argument('--maskperc', type=int, default=10, help='1,5,10,20,30,40,50')  # [**]
    # predict config
    parser.add_argument('--test_batch_size', type=int, default=8, help='test batch size ')
    # save
    parser.add_argument('--save_real', type=bool, default=False, help='save the real')
    return parser.parse_args()


if __name__ == '__main__':
    args_test = args_config()
