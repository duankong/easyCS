import numpy as np

def get_time(time):
    time=np.array(time).astype(np.int64)
    # print(time)
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    time_log="{:02d}:{:02d}:{:02d}".format(h,m,s)
    return time_log


if __name__ == '__main__':
    time=6666666.3
    print(get_time(time))
