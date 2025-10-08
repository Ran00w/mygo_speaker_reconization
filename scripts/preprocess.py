import os
import argparse
from utils.data_utils import split_data2


def get_args():
    """
    获取命令行参数
    
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--save_path', type=str, default='model.ckpt', help='模型保存路径')
    parser.add_argument('--steps', type=int, default=5000, help='训练步数')
    parser.add_argument('--n_spk', type=int, default=5, help='说话人数量')
    parser.add_argument('--warmup_steps', type=int, default=100, help='预热步数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--valid_steps', type=int, default=50, help='验证步数')
    parser.add_argument('--split_data', action='store_true', help='是否分割数据')
    return parser.parse_args()


def prepare_data():
    """
    准备数据集
    """
    # 如果数据集文件不存在，则创建
    if not os.path.exists('data/train.tsv'):
        print("正在分割数据集...")
        split_data2()
        print("数据集分割完成!")
    else:
        print("数据集已存在，跳过分割步骤")


if __name__ == "__main__":
    args = get_args()
    if args.split_data:
        prepare_data()