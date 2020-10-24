# -*- coding:utf-8 -*-
# copyright@zhanggugu
import six
import sys
import numpy as np
from data import buildHuffmanTree, HuffTree
# from huffman import buildHuffmanTree,HuffTree


def write_int(num, file_data):
    a4 = num & 255
    num = num >> 8
    a3 = num & 255
    num = num >> 8
    a2 = num & 255
    num = num >> 8
    a1 = num & 255
    file_data.write(six.int2byte(a1))
    file_data.write(six.int2byte(a2))
    file_data.write(six.int2byte(a3))
    file_data.write(six.int2byte(a4))
    return True


def read_int(file_data, position):
    a1 = file_data[position]
    a2 = file_data[position + 1]
    a3 = file_data[position + 2]
    a4 = file_data[position + 3]
    j = 0
    j = j | a1
    j = j << 8
    j = j | a2
    j = j << 8
    j = j | a3
    j = j << 8
    j = j | a4
    return j


def read_two_bytes(file_data, position):
    a1 = file_data[position]
    a2 = file_data[position + 1]
    j = 0
    j = j | a1
    # j = j << 8
    # j = j | a2
    return j


def compress(data, outputfilename):
    """
    压缩文件，参数有
    inputfilename：被压缩的文件的地址和名字
    outputfilename：压缩文件的存放地址和名字
    """
    # 1. 以二进制的方式打开文件
    # data=np.random.randint(0, 10, (4, 3))
    # data = np.array([1,2,3,7,7,7,7,7,7,7,7,255])
    data = data.reshape(-1)

    # 2. 统计 byte的取值［0-255］ 的每个值出现的频率
    # 保存在字典 char_freq中
    char_freq = {}
    for x in np.nditer(data):
        # tem = six.byte2int(filedata[x]) #python2.7 version
        tem = six.int2byte(x)  # python3.0 version
        if tem in char_freq.keys():
            char_freq[tem] = char_freq[tem] + 1
        else:
            char_freq[tem] = 1

    # 输出统计结果
    for tem in char_freq.keys():
        print(tem, ' : ', char_freq[tem])

    # 3. 开始构造原始的huffman编码树 数组，用于构造Huffman编码树
    list_hufftrees = []
    for x in char_freq.keys():
        # 使用 HuffTree的构造函数 定义一棵只包含一个叶节点的Huffman树
        tem = HuffTree(0, x, char_freq[x], None, None)
        # 将其添加到数组 list_hufftrees 当中
        list_hufftrees.append(tem)

    # 4. 步骤2中获取到的 每个值出现的频率的信息
    # 4.1. 保存叶节点的个数
    length = len(char_freq.keys())
    output = open(outputfilename, 'wb')

    # 一个int型的数有四个字节，所以将其分成四个字节写入到输出文件当中
    write_int(length, output)

    # 4.2  每个值 及其出现的频率的信息
    # 遍历字典 char_freq
    for x in char_freq.keys():
        # 写入data 一字节
        output.write(x)
        # 读取次数
        temp = char_freq[x]
        # 分成四个字节写入到压缩文件当中
        write_int(num=temp, file_data=output)
    output.close()
    # 5. 构造huffman编码树，并且获取到每个字符对应的 编码
    tem = buildHuffmanTree(list_hufftrees)
    tem.traverse_huffman_tree(tem.get_root(), '', char_freq)

    # 6. 开始对文件进行压缩
    output = open(outputfilename, 'ab')
    code = ''
    for element in np.nditer(data):
        # key = six.byte2int(filedata[i]) #python2.7 version
        key = six.int2byte(element)  # python3 version
        code = code + char_freq[key]
        out = 0
        while len(code) > 8:
            for x in range(8):
                out = out << 1
                if code[x] == '1':
                    out = out | 1
            code = code[8:]
            output.write(six.int2byte(out))
            out = 0

    # 处理剩下来的不满8位的code
    output.write(six.int2byte(len(code)))
    out = 0
    for i in range(len(code)):
        out = out << 1
        if code[i] == '1':
            out = out | 1
    for i in range(8 - len(code)):
        out = out << 1
    # 把最后一位给写入到文件当中
    output.write(six.int2byte(out))

    # 6. 关闭输出文件，文件压缩完毕
    output.close()


def decompress(inputfilename, outputfilename):
    """
    解压缩文件，参数有
    inputfilename：压缩文件的地址和名字
    outputfilename：解压缩文件的存放地址和名字
    """
    # 读取文件
    f = open(inputfilename, 'rb')
    filedata = f.read()
    # 获取文件的字节总数
    filesize = f.tell()
    # 1. 读取压缩文件中保存的树的叶节点的个数
    # 一下读取 4个 字节，代表一个int型的值
    j = read_int(file_data=filedata, position=0)
    leaf_node_size = j

    # 2. 读取压缩文件中保存的相信的原文件中 ［0-255］出现的频率的信息
    # 构造一个 字典 char_freq 一遍重建 Huffman编码树
    char_freq = {}
    for i in range(leaf_node_size):
        # c = six.byte2int(filedata[4+i*5+0]) # python2.7 version
        # c = filedata[4 + i * 5 + 0]  # python3 vesion
        c = read_two_bytes(file_data=filedata, position=4 + i * 5 + 0)
        # 同样的，出现的频率是int型的，读区四个字节来读取到正确的数值
        # python3
        j = read_int(file_data=filedata, position=4 + i * 5 + 1)
        print(c, j)
        char_freq[six.int2byte(c)] = j

    # 3. 重建huffman 编码树，和压缩文件中建立Huffman编码树的方法一致
    list_hufftrees = []
    for x in char_freq.keys():
        tem = HuffTree(0, x, char_freq[x], None, None)
        list_hufftrees.append(tem)

    tem = buildHuffmanTree(list_hufftrees)
    tem.traverse_huffman_tree(tem.get_root(), '', char_freq)

    # 4. 使用步骤3中重建的huffman编码树，对压缩文件进行解压缩
    output = open(outputfilename, 'wb')
    code = ''
    currnode = tem.get_root()
    for x in range(leaf_node_size * 5 + 4, filesize):
        # python3
        c = filedata[x]
        for i in range(8):
            if c & 128:
                code = code + '1'
            else:
                code = code + '0'
            c = c << 1

        while len(code) > 24:
            if currnode.isleaf():
                tem_byte = six.byte2int(currnode.get_value())
                output.write(bytes(str(tem_byte), encoding='utf-8'))
                output.write(bytes(six.int2byte(32)))
                currnode = tem.get_root()

            if code[0] == '1':
                currnode = currnode.get_right()
            else:
                currnode = currnode.get_left()
            code = code[1:]

    # 4.1 处理最后 24位
    sub_code = code[-16:-8]
    last_length = 0
    for i in range(8):
        last_length = last_length << 1
        if sub_code[i] == '1':
            last_length = last_length | 1

    code = code[:-16] + code[-8:-8 + last_length]

    while len(code) > 0:
        if currnode.isleaf():
            tem_byte = six.byte2int(currnode.get_value())
            output.write(bytes(str(tem_byte), encoding='utf-8'))
            output.write(bytes(six.int2byte(32)))
            currnode = tem.get_root()

        if code[0] == '1':
            currnode = currnode.get_right()
        else:
            currnode = currnode.get_left()
        code = code[1:]

    if currnode.isleaf():
        tem_byte = six.byte2int(currnode.get_value())
        output.write(bytes(str(tem_byte), encoding='utf-8'))
        output.write(bytes(six.int2byte(32)))
        currnode = tem.get_root()

    # 4. 关闭文件，解压缩完毕
    output.close()


if __name__ == '__main__':
    # 1. 获取用户的输入
    # FLAG 0 代表压缩文件 1代表解压缩文件
    # INPUTFILE： 输入的文件名
    # OUTPUTFILE：输出的文件名

    FLAG = '0'
    INPUTFILE = 'E:/Desktop/1.txt'
    OUTPUTFILE = 'E:/Desktop/flag.bin'
    Decompress = 'E:/Desktop/flag.txt'

    # 压缩文件
    data = np.array([1, 2, 3, 7, 7, 7, 7, 7, 7, 7, 56, 8, 7, 255])
    print('compress file')
    compress(data=data, outputfilename=OUTPUTFILE)
    # 解压缩文件
    print('decompress file')
    decompress(OUTPUTFILE, Decompress)
    print('[*] Done!')    
