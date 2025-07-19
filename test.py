import os
from dataset import audiodataset
from torch.utils.data import DataLoader
import torch
import torch .optim as optim
import torch.nn as nn
import numpy as np
from config import get_args
from model import Classifier, model_fn, valid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    test_data = audiodataset('data/test.tsv')
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    args = get_args()
    model = Classifier(n_spks=5)

    model.load_state_dict(torch.load(args.save_path)) 
    model.to(device)
    criterion = nn.CrossEntropyLoss()   
    acc = 0
    for i, batch in enumerate(test_loader):
        running_acc=0
        with torch.no_grad():
            loss, batch_acc = model_fn(batch, model, criterion, device)
            running_acc = batch_acc.item()
        acc += running_acc * batch[0].shape[0] / 100
    print("test acc: %.4f" % (acc))