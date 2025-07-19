import os
from dataset import audiodataset
from torch.utils.data import DataLoader
import torch
import torch .optim as optim
import torch.nn as nn
import sys
import numpy as np
from config import get_args
from model import Classifier, model_fn, valid
from transformers import get_cosine_schedule_with_warmup
from scipy.io import wavfile as wav

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    args = get_args()
    # fs, signal = wav.read('data/Anon/anon_part1.wav')  # 读取音频文件
    # print(fs)
    # print(fs*0.05)
    # sys.exit()
    train_data = audiodataset('data/train.tsv')
    valid_data = audiodataset('data/dev.tsv')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    model = Classifier()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.steps)
    criterion = nn.CrossEntropyLoss()   
    iterator = iter(train_loader)
    best_acc = 0
    best_state_dict = None
    model.to(device)
    model.train()
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
        print("step: %d, loss: %.4f, acc: %.4f" % (step, batch_loss, batch_acc))
        if step % args.valid_steps == 0:
            valid_acc = valid(valid_loader, model, criterion, device)
            print("valid acc: %.4f" % (valid_acc))
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_state_dict = model.state_dict()
                torch.save(best_state_dict, args.save_path)
                print("save model")
        
