import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import matplotlib.pyplot as plt
import numpy as np
import os

from config import Config
from data_utils import get_data_loaders
from model import Transformer
from utils import save_model, plot_training_results, ablation_study


class Trainer:
    def __init__(self, model, train_loader, val_loader, de_vocab, en_vocab):
        self.config = Config()
        self.model = model.to(self.config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab

        self.criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.lr,
                                    betas=self.config.betas, eps=self.config.eps)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

        self.train_losses = []
        self.val_losses = []
        self.accuracies = []

    # 在train_epoch方法中修改mask生成部分：
    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            src = batch['src'].to(self.config.device)
            tgt = batch['tgt'].to(self.config.device)

            # 准备输入和目标
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # 生成mask - 确保形状正确
            src_mask, tgt_mask = self.model.generate_mask(src, tgt_input)

            # 确保mask形状正确
            src_mask = src_mask.to(self.config.device)
            tgt_mask = tgt_mask.to(self.config.device)

            self.optimizer.zero_grad()

            # 前向传播
            output = self.model(src, tgt_input, src_mask, tgt_mask)

            # 计算损失
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch['src'].to(self.config.device)
                tgt = batch['tgt'].to(self.config.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask, tgt_mask = self.model.generate_mask(src, tgt_input)

                output = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

                total_loss += loss.item()

                # 计算准确率
                predictions = output.argmax(dim=-1)
                correct = (predictions == tgt_output) & (tgt_output != Config.PAD_IDX)
                total_correct += correct.sum().item()
                total_tokens += (tgt_output != Config.PAD_IDX).sum().item()

        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        return total_loss / len(self.val_loader), accuracy

    def train(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        print("开始训练Transformer模型...")
        print(f"设备: {self.config.device}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print(f"德语词汇表大小: {len(self.de_vocab)}")
        print(f"英语词汇表大小: {len(self.en_vocab)}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_loss, accuracy = self.validate()

            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.accuracies.append(accuracy)

            epoch_time = time.time() - start_time

            print(f'Epoch {epoch + 1:02d}/{num_epochs} | '
                  f'Time: {epoch_time:.2f}s | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Accuracy: {accuracy:.4f}')

            # 每10个epoch保存一次模型
            if (epoch + 1) % 10 == 0:
                save_model(self.model, self.de_vocab, self.en_vocab, epoch + 1)

    def run_ablation_study(self):
        print("\n开始消融实验...")
        ablation_results = ablation_study(self.de_vocab, self.en_vocab)
        return ablation_results

    def greedy_decode(self, src, max_length=50):
        """贪心解码生成翻译"""
        batch_size = src.size(0)

        # 编码源序列
        src_mask, _ = self.model.generate_mask(src, src)
        encoder_output = self.model.encoder(src, src_mask)

        # 初始化目标序列（只有开始标记）
        tgt = torch.ones(batch_size, 1).fill_(Config.SOS_IDX).long().to(self.config.device)

        for i in range(max_length - 1):
            # 生成mask
            _, tgt_mask = self.model.generate_mask(src, tgt)

            # 解码
            output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)
            output = self.model.output_layer(output)

            # 获取下一个词（贪心选择）
            next_word = output[:, -1:].argmax(dim=-1)
            tgt = torch.cat([tgt, next_word], dim=1)

            # 如果所有序列都生成了结束标记，提前停止
            if (next_word == Config.EOS_IDX).all():
                break

        return tgt
def hyperparameter_analysis():
    """只运行超参敏感性分析"""
    from hyperparameter_analysis import run_hyperparameter_analysis
    return run_hyperparameter_analysis()

def main():
    # 最简单的种子设置
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    # 创建结果目录
    os.makedirs('../results', exist_ok=True)

    # 加载数据
    train_loader, val_loader, de_vocab, en_vocab = get_data_loaders()

    # 创建模型
    config = Config()
    model = Transformer(
        src_vocab_size=len(de_vocab),
        tgt_vocab_size=len(en_vocab),
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_length=config.max_length,
        dropout=config.dropout
    )

    # 创建训练器并开始训练
    trainer = Trainer(model, train_loader, val_loader, de_vocab, en_vocab)
    trainer.train()


    # 绘制训练结果
    plot_training_results(trainer.train_losses, trainer.val_losses, trainer.accuracies)

    # 保存最终模型
    save_model(model, de_vocab, en_vocab, 'final')

    # 进行消融实验
    trainer.run_ablation_study()

    print("\n开始超参敏感性分析...")
    from hyperparameter_analysis import run_hyperparameter_analysis
    hyperparameter_results = run_hyperparameter_analysis()
    print("\n所有实验完成！")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--hyperparam':
        hyperparameter_analysis()
    else:
        main()