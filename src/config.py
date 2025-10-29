import torch


class Config:
    # 数据配置
    data_path = "../data/multi30k"
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