import time
from config import args_config
import numpy as np


def get_para_log(args_):
    log = "=" * 40 + "\n"
    log += "[*] Time is " + time.asctime(time.localtime(time.time())) + "\n"
    log += "\n[*] Model\n\n"
    log += "Model name:\t{}\n".format(args_.model)
    log += "Model save:\t{}\n".format(args_.model_checkpoint)
    log += "Model log :\t{}\n".format(args_.model_log)
    log += "Model show    ={} || save   ={}\n".format(args_.model_show, args_.model_save)
    log += "Model channels={:1d} || classes={:1d}\n".format(args_.img_n_channels, args_.img_n_classes)
    log += "\n[*] Data\n\n"
    log += "Path      :\t{}\n".format(args_.data_path)
    log += "Image Num :\t{:5d} /{:5d}\n".format(args_.data_star_num, args_.data_end_num)
    log += "Image Size:\t{:5d} /{:5d}\n".format(args_.img_size_x, args_.img_size_y)
    log += "\n[*] Mask\n\n"
    log += "Mask info :\t{} -- {}\n".format(args_.maskname, args_.maskperc)
    log += "\n[*] Loss\n\n"
    log += "MSE_only={}\tSSIM={}\tVGG={}\n".format(args_.loss_mse_only, args_.loss_ssim, args_.loss_vgg)
    log += "alpha(MSE)={}\tgamma(SSIM)={}\tbeta(VGG)={}\n".format(args_.alpha, args_.gamma, args_.beta)
    log += "\n[*] Train\n\n"
    log += "epochs={}\tlr={}\tbatch_size={}\ttest_model={}\n\n".format(args_.epochs, args_.lr, args_.batch_size,
                                                                       args_.test_model)
    if args_.model == "DQBCS":
        log += "\n[*] DQBCS para\n\n"
        log += "Net sample rate={}\n".format(args_.DQBCS_rate)
    return log


if __name__ == "__main__":
    args_ = args_config()
    print(get_para_log(args_))
