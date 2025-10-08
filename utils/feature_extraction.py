import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
import random
from torch.utils.data import Dataset
import torch


def get_mfcc(data, fs):
    """
    提取MFCC特征
    
    Args:
        data: 音频数据
        fs: 采样率
        
    Returns:
        feature: MFCC特征
    """
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
    if feature.shape[0] > 96:
        feature = feature[:96, :]
    else:
        feature = np.pad(feature, ((0, 96-feature.shape[0]), (0, 0)), 'constant')
    
    return feature


def load_audio(datas, audio_file, label):
    """
    加载音频文件
    
    Args:
        datas: 数据列表
        audio_file: 音频文件路径
        label: 标签
        
    Returns:
        datas: 更新后的数据列表
    """
    fs, signal = wav.read(audio_file)  # 读取音频文件
    feature = get_mfcc(signal, fs)  # 获取MFCC特征
    datas.append([feature, label])
    return datas


def load_wav(audios, fs):
    """
    加载WAV文件
    
    Args:
        audios: 音频列表
        fs: 采样率
        
    Returns:
        datas: 数据列表
    """
    datas = []
    for audio in audios:
        signal = audio
        feature = get_mfcc(signal, fs)
        datas.append([feature, 0])
    return datas


def loader(tsv, batch_size, is_random=True):
    """
    数据读取器
    
    Args:
        tsv: TSV文件路径
        batch_size: 批次大小
        is_random: 是否随机打乱
        
    Yields:
        features: 特征
        labels: 标签
    """
    datas = []
    with open(tsv, 'r', encoding='UTF-8') as f:
        for line in f:
            audio, label = line[:-1].split('\t')            
            fs, signal = wav.read('data/'+audio) # 读取音频文件
            feature = get_mfcc(signal, fs)
            datas.append([feature, int(label)])
    
    features = []
    labels = []
    if is_random:
        random.shuffle(datas)
    for data in datas:
        feature, label = data
        features.append(feature)
        labels.append(label)
        if len(labels) == batch_size:
            features = np.concatenate(features, 0).reshape(-1, 13, 3, 64).astype('float32')
            labels = np.array(labels).reshape(-1, 1).astype('int64')
            yield features, labels
            features = []
            labels = []