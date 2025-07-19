import torch
from model import Classifier
from torchsummary import torchsummary
import numpy as np
from config import get_args

def print_model_structure():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = Classifier(n_spks=5)
    
    # 如果有保存的模型权重，加载它
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.save_path))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(args.save_path, map_location=torch.device('cpu')))
    # 打印模型结构
    print("\n=== 模型结构 ===")
    print(model)
    
    # 使用torchsummary打印详细信息
    print("\n=== 详细参数信息 ===")
    # 注意：这里的input_size要根据实际输入维度调整
    torchsummary.summary(model, input_size=(1, 96, 40))
    
    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== 参数统计 ===")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / (1024*1024):.2f} MB")  # 假设每个参数是4字节

if __name__ == "__main__":
    print_model_structure()
