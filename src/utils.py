import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from model import Transformer
from data_utils import TranslationDataset, get_data_loaders
from config import Config


def save_model(model, de_vocab, en_vocab, epoch):
    """保存模型和词汇表"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'de_vocab': de_vocab,
        'en_vocab': en_vocab,
        'epoch': epoch,
        'config': {
            'd_model': Config.d_model,
            'num_layers': Config.num_layers,
            'num_heads': Config.num_heads,
            'd_ff': Config.d_ff,
            'max_length': Config.max_length
        }
    }

    filename = f"../results/transformer_epoch_{epoch}.pth"
    torch.save(checkpoint, filename)
    print(f"模型已保存: {filename}")


def plot_training_results(train_losses, val_losses, accuracies):
    """绘制训练结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(accuracies, label='Accuracy', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def ablation_study(de_vocab, en_vocab):
    """消融实验"""
    config = Config()
    train_loader, val_loader, _, _ = get_data_loaders()

    experiments = {
        'baseline': {'num_layers': 3, 'num_heads': 8, 'd_model': 256, 'd_ff': 512},
        'no_pos_encoding': {'num_layers': 3, 'num_heads': 8, 'd_model': 256, 'd_ff': 512, 'no_pos_encoding': True},
        'single_head': {'num_layers': 3, 'num_heads': 1, 'd_model': 256, 'd_ff': 512},
        'single_layer': {'num_layers': 1, 'num_heads': 8, 'd_model': 256, 'd_ff': 512},
        'small_ffn': {'num_layers': 3, 'num_heads': 8, 'd_model': 256, 'd_ff': 128},
    }

    results = {}

    for exp_name, exp_config in experiments.items():
        print(f"\n正在进行消融实验: {exp_name}")
        print(f"配置: {exp_config}")

        # 创建模型
        model = Transformer(
            src_vocab_size=len(de_vocab),
            tgt_vocab_size=len(en_vocab),
            d_model=exp_config['d_model'],
            num_layers=exp_config['num_layers'],
            num_heads=exp_config['num_heads'],
            d_ff=exp_config['d_ff'],
            max_length=config.max_length
        )

        # 移除位置编码
        if exp_config.get('no_pos_encoding', False):
            model.encoder.pos_encoding = nn.Identity()
            model.decoder.pos_encoding = nn.Identity()

        # 训练（简化版，只训练少量epoch用于比较）
        from train import Trainer
        trainer = Trainer(model, train_loader, val_loader, de_vocab, en_vocab)
        trainer.train(num_epochs=10)

        # 记录结果
        results[exp_name] = {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'accuracies': trainer.accuracies,
            'final_val_loss': trainer.val_losses[-1],
            'final_accuracy': trainer.accuracies[-1]
        }

    # 绘制消融实验结果
    plot_ablation_results(results)
    return results


def plot_ablation_results(results):
    """绘制消融实验结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 验证损失比较
    for exp_name, result in results.items():
        ax1.plot(result['val_losses'], label=exp_name, linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Ablation Study - Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率比较
    for exp_name, result in results.items():
        ax2.plot(result['accuracies'], label=exp_name, linewidth=2)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Ablation Study - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/ablation_study.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印最终结果比较
    print("\n消融实验最终结果:")
    print("=" * 60)
    for exp_name, result in results.items():
        print(f"{exp_name:20} | Val Loss: {result['final_val_loss']:.4f} | Accuracy: {result['final_accuracy']:.4f}")


def load_model_for_inference(model_path):
    """加载模型用于推理"""
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']

    model = Transformer(
        src_vocab_size=len(checkpoint['de_vocab']),
        tgt_vocab_size=len(checkpoint['en_vocab']),
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_length=config['max_length']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint['de_vocab'], checkpoint['en_vocab']