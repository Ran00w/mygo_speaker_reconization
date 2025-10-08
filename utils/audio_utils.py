import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
import random
import webrtcvad
from pydub import AudioSegment
from collections import Counter


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


def is_silent(signal, threshold=80):
    """
    判断信号是否为空白声
    
    Args:
        signal: 音频信号数组
        threshold: 判断空白声的振幅阈值，默认值为80
        
    Returns:
        bool: 如果信号为空白声，返回True；否则返回False
    """
    # 计算信号的绝对值并检查是否低于阈值
    return np.average(np.abs(signal)) <= threshold


def split_audio(input_dir, input_filename, output_dir, segment_length=1000, window_length=10):
    """
    分割音频文件
    
    Args:
        input_dir: 输入目录
        input_filename: 输入文件名
        output_dir: 输出目录
        segment_length: 分割长度，默认1000ms
        window_length: 窗口长度，默认10
        
    Returns:
        silent_list: 静音片段列表
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    audio = AudioSegment.from_wav(os.path.join(input_dir, input_filename))
    duration = len(audio)
    base_name = os.path.splitext(input_filename)[0]
    
    i = 0  
    start = 0
    silent_list = []
    while start < duration:
        start = i * segment_length/window_length
        end = min(start + segment_length, duration)
        segment = audio[start:end]
        samples = np.array(segment.get_array_of_samples())
        if samples.size == 0:
            continue
            
        if is_silent(samples) == False:
            samples = samples / np.max(np.abs(samples))*100
            segment = segment._spawn(samples.astype(np.int16).tobytes())
        else:
            silent_list.append(i+1)
        out_name = f"{base_name}_part{i+1}.wav"
        segment.export(os.path.join(output_dir, out_name), format="wav")
        i += 1
    return silent_list


def VAD(audio, mode):
    """
    语音活动检测
    
    Args:
        audio: 音频文件路径
        mode: VAD模式
        
    Returns:
        audios: 分割后的音频列表
        fs: 采样率
    """
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


def jud(result):
    """
    判断最常见的结果
    
    Args:
        result: 结果列表
        
    Returns:
        最常见的结果
    """
    if not result:
        return None
    counter = Counter(result)
    return counter.most_common(1)[0][0]