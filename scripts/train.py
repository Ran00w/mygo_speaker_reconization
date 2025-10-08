import os
import sys
sys.path.append('..')

from datasets.dataset import AudioDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from models.classifier import Classifier, model_fn, valid
from transformers import get_cosine_schedule_with_warmup
from scripts.preprocess import get_args, prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = get_args()
    
    # 准备数据
    prepare_data()
    
    # 加载数据集
    train_data = AudioDataset('data/train.tsv')
    valid_data = AudioDataset('data/dev.tsv')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    
    # 初始化模型
    model = Classifier(n_spks=args.n_spk)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.steps)
    criterion = nn.CrossEntropyLoss()   
    
    iterator = iter(train_loader)
    best_acc = 0
    best_state_dict = None
    model.to(device)
    model.train()
    
    print(f"开始训练，设备: {device}")
    print(f"训练参数: batch_size={args.batch_size}, learning_rate={args.learning_rate}, steps={args.steps}")
    
    for step in range(args.steps):
        try: 
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
            
        loss, acc = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_acc = acc.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        print(f"step: {step}, loss: {batch_loss:.4f}, acc: {batch_acc:.4f}")
        
        if step % args.valid_steps == 0:
            valid_acc = valid(valid_loader, model, criterion, device)
            print(f"valid acc: {valid_acc:.4f}")
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_state_dict = model.state_dict()
                torch.save(best_state_dict, args.save_path)
                print(f"保存模型，验证准确率: {best_acc:.4f}")
    
    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")
        
