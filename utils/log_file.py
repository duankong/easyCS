import time
from config import args_config


def get_para_log(args_):
    log = "=" * 40+"\n"
    log += "[*] Time is " + time.asctime(time.localtime(time.time())) + "\n\n"
    log += "[*] Data\n\n"
    log += "Path      :\t{}\n".format(args_.data_path)
    log += "Image Num :\t{: 5d} /{:5d}\n".format(args_.data_star_num, args_.data_end_num)
    log += "Image Size:\t{: 5d} /{:5d}\n".format(args_.img_size_x, args_.img_size_y)
    log += "Mask info :\t{} -- {}\n".format(args_.maskname, args_.maskperc)
    log += "\n[*] Model\n\n"
    log += "Model name:\t{}\n".format(args_.model)
    log += "Model save:\t{}\n".format(args_.model_name)
    log += "Model log :\t{}\n".format(args_.model_log)
    log += "Model show    ={:2d} || save   ={:2d}\n".format(args_.model_show, args_.model_save)
    log += "Model channels={:2d} || classes={:2d}\n".format(args_.img_n_channels, args_.img_n_classes)
    log += "\n[*] Train\n\n"
    log += "epochs={}\tlr={}\tbatch_size={}\n".format(args_.epochs, args_.lr, args_.batch_size)
    log += "alpha={}\tgamma={}\tbeta={}\n\n".format(args_.alpha, args_.gamma, args_.beta)


    return log


if __name__ == "__main__":
    args_ = args_config()
    print(get_para_log(args_))
