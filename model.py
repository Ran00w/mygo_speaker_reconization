import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
  def __init__(self, d_model=80, n_spks=5, dropout=0.1):
    super().__init__()
    # Project the dimension of features from that of input into d_model.
    self.prenet = nn.Linear(40, d_model)
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, dim_feedforward=256, nhead=2
    )
    self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    # Project the the dimension of features from d_model into speaker nums.
    self.pred_layer = nn.Sequential(
      nn.Linear(d_model, n_spks),
    )

  def forward(self, mels):
    """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
    # out: (batch size, length, d_model)
    out = self.prenet(mels)
    # out: (length, batch size, d_model)
    out = out.permute(1, 0, 2)
    # The encoder layer expect features in the shape of (length, batch size, d_model).
    out = self.encoder(out)
    # out: (batch size, length, d_model)
    out = out.transpose(0, 1)
    # mean pooling
    stats = out.mean(dim=1)
    # out: (batch, n_spks)
    out = self.pred_layer(stats)
    return out

def model_fn(batch, model, criterion, device):
    mels, labels = batch
    mels.to(device)
    labels.to(device)
    out = model(mels)
    loss = criterion(out, labels)
    pred = out.argmax(1)
    accuracy = torch.mean((pred==labels).float())
    return loss, accuracy

def valid(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0
    running_accuracy = 0
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, acc = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += acc.item()
    return running_accuracy / len(dataloader)

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def predict(audio_path):
    # 模拟预测逻辑
    return ['Anon', 'Rana', 'Soyo', 'Taki', 'Tomori']

