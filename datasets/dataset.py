import os
from torch.utils.data import Dataset
from utils.feature_extraction import get_mfcc
import scipy.io.wavfile as wav
import torch


class AudioDataset(Dataset):
    """
    音频数据集类
    """
    def __init__(self, tsv_file):
        """
        初始化数据集
        
        Args:
            tsv_file: TSV文件路径
        """
        self.data = []
        with open(tsv_file, 'r', encoding='UTF-8') as f:
            for line in f:
                audio, label = line[:-1].split('\t')
                self.data.append((audio, int(label)))
    
    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            数据集大小
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        获取数据项
        
        Args:
            index: 索引
            
        Returns:
            feature: 特征
            label: 标签
        """
        audio, label = self.data[index]
        fs, signal = wav.read(os.path.join('data', audio)) 
        feature = get_mfcc(signal, fs)
        feature = torch.tensor(feature, dtype=torch.float32)  # 保证特征为float32的Tensor
        return feature, label