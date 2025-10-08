import os
import random


def split_data1():
    """
    数据划分函数1
    """
    # 读取所有音频数据
    recordings = ['recordings/'+_ for _ in os.listdir('work/recordings')]
    total = []
    for recording in recordings:
        label = int(recording[11])
        total.append('%s\t%s\n' % (recording, label))

    train = open('work/train.tsv', 'w', encoding='UTF-8')
    dev = open('work/dev.tsv', 'w', encoding='UTF-8')
    test = open('work/test.tsv', 'w', encoding='UTF-8')

    random.shuffle(total)
    split_num = int((len(total)-100)*0.9)
    for line in total[:split_num]:
        train.write(line)
    for line in total[split_num:-100]:
        dev.write(line)
    for line in total[-100:]:
        test.write(line)

    train.close()
    dev.close()
    test.close()


def split_data2():
    """
    数据划分函数2 - 用于MyGO角色数据
    """
    # 读取所有音频数据
    total = []
    name2num = {"Anon":0,"Rana":1,"Soyo":2,"Taki":3,"Tomori":4}
    for name in name2num.keys(): 
        dir_name = name  # 文件夹名称
        recordings = [dir_name + '/' + _ for _ in os.listdir('data/' + dir_name)]
        for recording in recordings:
            total.append('%s\t%s\n' % (recording, name2num[name]))  # 标签就是文件夹的名字

    train = open('data/train.tsv', 'w', encoding='UTF-8')
    dev = open('data/dev.tsv', 'w', encoding='UTF-8')
    test = open('data/test.tsv', 'w', encoding='UTF-8')

    random.shuffle(total)
    split_num = int((len(total)-100)*0.9)
    for line in total[:split_num]:
        train.write(line)
    for line in total[split_num:-100]:
        dev.write(line)
    for line in total[-100:]:
        test.write(line)

    train.close()
    dev.close()
    test.close()


def split_data2_without_rana():
    """
    数据划分函数3 - 不包含Rana的数据
    """
    # 读取所有音频数据
    total = []
    name2num = {"Anon":0,"Rana":1,"Soyo":2,"Taki":3,"Tomori":4}
    for name in name2num.keys(): 
        if name == "Rana":
            continue
        dir_name = name  # 文件夹名称
        recordings = [dir_name + '/' + _ for _ in os.listdir('data/' + dir_name)]
        for recording in recordings:
            total.append('%s\t%s\n' % (recording, name2num[name]))  # 标签就是文件夹的名字

    train = open('data/train.tsv', 'w', encoding='UTF-8')
    dev = open('data/dev.tsv', 'w', encoding='UTF-8')
    test = open('data/test.tsv', 'w', encoding='UTF-8')

    random.shuffle(total)
    split_num = int((len(total)-100)*0.9)
    for line in total[:split_num]:
        train.write(line)
    for line in total[split_num:-100]:
        dev.write(line)
    for line in total[-100:]:
        test.write(line)

    train.close()
    dev.close()
    test.close()


def split_data():
    """
    数据划分函数 - 通用版本
    """
    # 读取部分
    total = []
    for i in range(10):  # 对于每个数字
        dir_name = str(i)  # 文件夹名称
        recordings = [dir_name + '/' + _ for _ in os.listdir('data/' + dir_name)[:2000]]  # 只读取前2000个文件
        for recording in recordings:
            total.append('%s\t%s\n' % (recording, i))  # 标签就是文件夹的名字

    train = open('data/train.tsv', 'w', encoding='UTF-8')
    dev = open('data/dev.tsv', 'w', encoding='UTF-8')
    test = open('data/test.tsv', 'w', encoding='UTF-8')

    random.shuffle(total)
    split_num = int((len(total)-1000)*0.9)
    for line in total[:split_num]:
        train.write(line)
    for line in total[split_num:-1000]:
        dev.write(line)
    for line in total[-1000:]:
        test.write(line)

    train.close()
    dev.close()
    test.close()