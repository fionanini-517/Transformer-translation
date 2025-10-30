import os

import torch


class Config:
    # 数据配置
    # 获取当前脚本（config.py）的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接数据路径（假设data在src目录同级，若在src内则改为"data/multi30k"）
    data_path = os.path.join(current_dir, "../data/multi30k")  # 根据实际位置调整
    # data_path2="../data/multi30k"
    max_length = 50
    batch_size = 32

    # 模型配置 - 调整为更小的配置以便调试
    d_model = 128
    num_heads = 8
    num_layers = 1
    d_ff = 512
    dropout = 0.1

    # 训练配置
    lr = 0.001
    num_epochs = 50
    betas = (0.9, 0.98)
    eps = 1e-9
    grad_clip = 1.0

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 词汇表特殊标记
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3