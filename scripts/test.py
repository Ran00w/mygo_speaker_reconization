import os
import sys
sys.path.append('..')

from datasets.dataset import AudioDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from models.classifier import Classifier, model_fn, valid
from scripts.preprocess import get_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = get_args()
    
    # 加载测试数据集
    test_data = AudioDataset('data/test.tsv')
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    
    # 初始化模型
    model = Classifier(n_spks=args.n_spk)
    model.load_state_dict(torch.load(args.save_path)) 
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()   
    acc = 0
    total_samples = 0
    
    print(f"开始测试，设备: {device}")
    print(f"测试模型: {args.save_path}")
    
    for i, batch in enumerate(test_loader):
        running_acc = 0
        with torch.no_grad():
            loss, batch_acc = model_fn(batch, model, criterion, device)
            running_acc = batch_acc.item()
        
        batch_size = batch[0].shape[0]
        acc += running_acc * batch_size
        total_samples += batch_size
        
        print(f"批次 {i+1}: 准确率 = {running_acc:.4f}, 样本数 = {batch_size}")
    
    test_acc = acc / total_samples
    print(f"测试完成，总体准确率: {test_acc:.4f}")