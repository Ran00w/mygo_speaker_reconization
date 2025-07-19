# utils.py

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
import random
from torch.utils.data import Dataset
import webrtcvad
from pydub import AudioSegment
from collections import Counter


# MFCC特征提取
def get_mfcc(data, fs):
    # 计算帧长
    winlen = 0.05 
    if len(data.shape) == 2: # 如果是双声道音频
        data = (data[:, 0]+data[:, 1])/2
    frame_length = int(fs * winlen)
    # 取大于等于frame_length的最小2的幂作为NFFT
    NFFT = 2 ** int(np.ceil(np.log2(frame_length)))
    # MFCC特征提取，显式指定NFFT
    
    # 特征重塑

    feature = mfcc(data, fs, numcep=40, winlen=winlen, nfft=NFFT, nfilt=40)
    # 对数据进行截取或者填充
    if feature.shape[0]>96:
        feature = feature[:96, :]
    else:
        feature = np.pad(feature, ((0, 96-feature.shape[0]), (0, 0)), 'constant')
    # 通道转置(HWC->CHW)
    # 新建空维度(CHW->NCHW)
    
    return feature



def load_audio(datas,audio_file,label):
    fs, signal = wav.read(audio_file)  # 读取音频文件
    feature = get_mfcc(signal, fs)  # 获取MFCC特征
    datas.append([feature, label])
    return datas

def load_wav(audios,fs):
    datas=[]
    for audio in audios:
        signal = audio
        feature = get_mfcc(signal, fs)
        datas.append([feature, 0])
    return datas


# 读取数据列表文件
def loader(tsv):
    datas = []
    with open(tsv, 'r', encoding='UTF-8') as f:
        for line in f:
            audio, label = line[:-1].split('\t')            
            fs, signal = wav.read('data/'+audio) # 读取音频文件
            feature = get_mfcc(signal, fs)
            datas.append([feature, int(label)])
    return datas

# 数据读取器
def reader(datas, batch_size, is_random=True):
    features = []
    labels = []
    if is_random:
        random.shuffle(datas)
    for data in datas:
        feature, label = data
        features.append(feature)
        labels.append(label)
        if len(labels)==batch_size:
            features = np.concatenate(features, 0).reshape(-1, 13, 3, 64).astype('float32')
            labels = np.array(labels).reshape(-1, 1).astype('int64')
            yield features, labels
            features = []
            labels = []

# 数据划分函数
def split_data1():
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

# 数据划分函数
def split_data2():
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

def VAD(audio, mode):
    # 读取音频
    fs, signal = wav.read(audio)
    # vad初始化
    vad = webrtcvad.Vad()
    vad.set_mode(mode)
    # 数据填充
    padding = int(fs*0.02) - (signal.shape[0] % int(fs*0.02))
    if padding < 0:
        padding += int(fs*0.02)
    signal = np.pad(signal, (0, padding), 'constant')
    # 数据分帧
    lens = signal.shape[0]
    signals = np.split(signal, lens//int(fs*0.02))
    # 音频切分
    audio = []
    audios = []
    for signal in signals:
        if vad.is_speech(signal, fs):
            audio.append(signal)
        elif len(audio) and (not vad.is_speech(signal, fs)):
            audios.append(np.concatenate(audio, 0))
            audio = []
    return audios, fs

def split_audio(input_dir, input_filename, output_dir, segment_length=1000, window_length=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    audio = AudioSegment.from_wav(os.path.join(input_dir, input_filename))
    duration = len(audio)
    base_name = os.path.splitext(input_filename)[0]
    # for i, start in enumerate(range(0, duration, segment_length)):
    #     end = min(start + segment_length, duration)
    #     segment = audio[start:end]

    #     out_name = f"{base_name}_part{i+1}.wav"
    #     segment.export(os.path.join(output_dir, out_name), format="wav")
    i = 0  
    start = 0
    silent_list = []
    while start < duration:
        start = i * segment_length/window_length
        end = min(start + segment_length, duration)
        segment = audio[start:end]
        samples = np.array(segment.get_array_of_samples())
        if samples.size == 0:
#            print(f"跳过空白片段: {base_name}_part{i+1}.wav")
#            i += 1
            continue
#        if is_silent(samples) == False:
#        if is_silent(samples) == True:
#            print(i+1, "is silent")
        if is_silent(samples) == False:
            samples = samples / np.max(np.abs(samples))*100
            segment = segment._spawn(samples.astype(np.int16).tobytes())
        else:
            silent_list.append(i+1)
        out_name = f"{base_name}_part{i+1}.wav"
        segment.export(os.path.join(output_dir, out_name), format="wav")
        i += 1
    return silent_list


def is_silent(signal, threshold=80):
    """
    判断信号是否为空白声。

    参数：
        signal (numpy.ndarray): 音频信号数组。
        threshold (int): 判断空白声的振幅阈值，默认值为100。

    返回：
        bool: 如果信号为空白声，返回True；否则返回False。
    """
    # 计算信号的绝对值并检查是否低于阈值
    return np.average(np.abs(signal)) <= threshold

def jud(result):
    if not result:
        return None
    counter = Counter(result)
    return counter.most_common(1)[0][0] 
if __name__ == "__main__":
    split_data2_without_rana()