import os
from torch.utils.data import Dataset
from utils import get_mfcc
import scipy.io.wavfile as wav
import torch

class audiodataset(Dataset):
    def __init__(self, tsv_file):
        self.data = []
        with open(tsv_file, 'r', encoding='UTF-8') as f:
            for line in f:
                audio, label = line[:-1].split('\t')
                self.data.append((audio, int(label)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        audio, label = self.data[index]
        fs, signal = wav.read(os.path.join('data', audio)) 
        feature = get_mfcc(signal, fs)
        feature = torch.tensor(feature, dtype=torch.float32)  # 保证特征为float32的Tensor
        return feature, label

