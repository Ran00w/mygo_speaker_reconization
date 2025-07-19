import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--save_path', type=str, default='model.ckpt', help='')
    parser.add_argument('--steps',type=int, default=5000, help='')
    parser.add_argument('--n_spk',type=int, default=5, help='')
    parser.add_argument('--warmup_steps', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='')
    parser.add_argument('--valid_steps', type=int, default=50, help='')
    return parser.parse_args()   