import torch
import torch.nn as nn
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from model import Transformer
from data_utils import get_data_loaders
from config import Config

# 设置matplotlib不显示图形
plt.switch_backend('Agg')


class HyperparameterAnalyzer:
    def __init__(self, de_vocab, en_vocab):
        self.config = Config()
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab
        self.results = defaultdict(dict)

        # 使用绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.analysis_dir = os.path.join(current_dir, '../analysis_results')
        os.makedirs(self.analysis_dir, exist_ok=True)
        print(f"分析结果将保存到: {os.path.abspath(self.analysis_dir)}")

    def analyze_learning_rate(self, lr_values=None):
        """学习率敏感性分析"""
        if lr_values is None:
            lr_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

        print("正在进行学习率敏感性分析...")

        for lr in lr_values:
            print(f"\n测试学习率: {lr}")

            # 创建模型
            model = Transformer(
                src_vocab_size=len(self.de_vocab),
                tgt_vocab_size=len(self.en_vocab),
                d_model=self.config.d_model,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                d_ff=self.config.d_ff,
                max_length=self.config.max_length,
                dropout=self.config.dropout
            )

            # 训练并评估
            final_accuracy, final_loss, convergence_epoch = self._train_and_evaluate(model, lr=lr)

            self.results['learning_rate'][lr] = {
                'final_accuracy': final_accuracy,
                'final_loss': final_loss,
                'convergence_epoch': convergence_epoch
            }

        # 绘制并保存学习率分析图
        self._plot_learning_rate()
        return self.results['learning_rate']

    def analyze_model_dimension(self, d_model_values=None):
        """模型维度敏感性分析"""
        if d_model_values is None:
            d_model_values = [64, 128, 256, 512]

        print("正在进行模型维度敏感性分析...")

        for d_model in d_model_values:
            print(f"\n测试模型维度: {d_model}")

            # 确保头数整除模型维度
            num_heads = min(8, d_model // 8)  # 自适应头数

            model = Transformer(
                src_vocab_size=len(self.de_vocab),
                tgt_vocab_size=len(self.en_vocab),
                d_model=d_model,
                num_layers=self.config.num_layers,
                num_heads=num_heads,
                d_ff=d_model * 4,  # 按比例设置前馈网络维度
                max_length=self.config.max_length,
                dropout=self.config.dropout
            )

            final_accuracy, final_loss, convergence_epoch = self._train_and_evaluate(model)

            self.results['model_dimension'][d_model] = {
                'final_accuracy': final_accuracy,
                'final_loss': final_loss,
                'convergence_epoch': convergence_epoch,
                'parameters': sum(p.numel() for p in model.parameters())
            }

        # 绘制并保存模型维度分析图
        self._plot_model_dimension()
        return self.results['model_dimension']

    def analyze_attention_heads(self, num_heads_values=None):
        """注意力头数敏感性分析"""
        if num_heads_values is None:
            num_heads_values = [1, 2, 4, 8, 16]

        print("正在进行注意力头数敏感性分析...")

        for num_heads in num_heads_values:
            print(f"\n测试注意力头数: {num_heads}")

            # 确保头数整除模型维度
            d_model = max(64, num_heads * 8)  # 自适应模型维度

            model = Transformer(
                src_vocab_size=len(self.de_vocab),
                tgt_vocab_size=len(self.en_vocab),
                d_model=d_model,
                num_layers=self.config.num_layers,
                num_heads=num_heads,
                d_ff=self.config.d_ff,
                max_length=self.config.max_length,
                dropout=self.config.dropout
            )

            final_accuracy, final_loss, convergence_epoch = self._train_and_evaluate(model)

            self.results['attention_heads'][num_heads] = {
                'final_accuracy': final_accuracy,
                'final_loss': final_loss,
                'convergence_epoch': convergence_epoch
            }

        # 绘制并保存注意力头数分析图
        self._plot_attention_heads()
        return self.results['attention_heads']

    def analyze_layers_depth(self, num_layers_values=None):
        """网络深度敏感性分析"""
        if num_layers_values is None:
            num_layers_values = [1, 2, 3, 4, 6]

        print("正在进行网络深度敏感性分析...")

        for num_layers in num_layers_values:
            print(f"\n测试网络层数: {num_layers}")

            model = Transformer(
                src_vocab_size=len(self.de_vocab),
                tgt_vocab_size=len(self.en_vocab),
                d_model=self.config.d_model,
                num_layers=num_layers,
                num_heads=self.config.num_heads,
                d_ff=self.config.d_ff,
                max_length=self.config.max_length,
                dropout=self.config.dropout
            )

            final_accuracy, final_loss, convergence_epoch = self._train_and_evaluate(model)

            self.results['layers_depth'][num_layers] = {
                'final_accuracy': final_accuracy,
                'final_loss': final_loss,
                'convergence_epoch': convergence_epoch,
                'parameters': sum(p.numel() for p in model.parameters())
            }

        # 绘制并保存网络深度分析图
        self._plot_layers_depth()
        return self.results['layers_depth']

    def analyze_dropout_rate(self, dropout_values=None):
        """Dropout率敏感性分析"""
        if dropout_values is None:
            dropout_values = [0.0, 0.1, 0.2, 0.3, 0.5]

        print("正在进行Dropout率敏感性分析...")

        for dropout in dropout_values:
            print(f"\n测试Dropout率: {dropout}")

            model = Transformer(
                src_vocab_size=len(self.de_vocab),
                tgt_vocab_size=len(self.en_vocab),
                d_model=self.config.d_model,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                d_ff=self.config.d_ff,
                max_length=self.config.max_length,
                dropout=dropout
            )

            final_accuracy, final_loss, convergence_epoch = self._train_and_evaluate(model)

            self.results['dropout_rate'][dropout] = {
                'final_accuracy': final_accuracy,
                'final_loss': final_loss,
                'convergence_epoch': convergence_epoch
            }

        # 绘制并保存Dropout率分析图
        self._plot_dropout_rate()
        return self.results['dropout_rate']

    def analyze_batch_size(self, batch_size_values=None):
        """批大小敏感性分析"""
        if batch_size_values is None:
            batch_size_values = [16, 32, 64, 128]

        print("正在进行批大小敏感性分析...")

        original_batch_size = self.config.batch_size

        for batch_size in batch_size_values:
            print(f"\n测试批大小: {batch_size}")

            # 临时修改配置
            self.config.batch_size = batch_size
            train_loader, val_loader, _, _ = get_data_loaders()

            model = Transformer(
                src_vocab_size=len(self.de_vocab),
                tgt_vocab_size=len(self.en_vocab),
                d_model=self.config.d_model,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                d_ff=self.config.d_ff,
                max_length=self.config.max_length,
                dropout=self.config.dropout
            )

            final_accuracy, final_loss, convergence_epoch = self._train_and_evaluate(
                model, train_loader, val_loader
            )

            self.results['batch_size'][batch_size] = {
                'final_accuracy': final_accuracy,
                'final_loss': final_loss,
                'convergence_epoch': convergence_epoch
            }

        # 恢复原始批大小
        self.config.batch_size = original_batch_size

        # 绘制并保存批大小分析图
        self._plot_batch_size()
        return self.results['batch_size']

    def _train_and_evaluate(self, model, train_loader=None, val_loader=None, lr=None, num_epochs=10):
        """训练并评估模型"""
        from train import Trainer

        if train_loader is None or val_loader is None:
            train_loader, val_loader, _, _ = get_data_loaders()

        # 创建训练器
        trainer = Trainer(model, train_loader, val_loader, self.de_vocab, self.en_vocab)

        # 如果指定了学习率，修改优化器
        if lr is not None:
            trainer.optimizer = torch.optim.Adam(
                model.parameters(), lr=lr,
                betas=self.config.betas, eps=self.config.eps
            )

        # 简化训练
        trainer.train(num_epochs=num_epochs)

        # 获取最终结果
        final_accuracy = trainer.accuracies[-1] if trainer.accuracies else 0
        final_loss = trainer.val_losses[-1] if trainer.val_losses else float('inf')

        # 计算收敛轮次（损失下降小于阈值的第一个轮次）
        convergence_epoch = self._calculate_convergence_epoch(trainer.val_losses)

        return final_accuracy, final_loss, convergence_epoch

    def _calculate_convergence_epoch(self, losses, threshold=0.001):
        """计算模型收敛的轮次"""
        if len(losses) < 3:
            return len(losses)

        for i in range(2, len(losses)):
            if abs(losses[i] - losses[i - 1]) < threshold and abs(losses[i - 1] - losses[i - 2]) < threshold:
                return i
        return len(losses)

    def run_comprehensive_analysis(self, analysis_types=None):
        """运行全面的超参分析"""
        if analysis_types is None:
            analysis_types = [
                'learning_rate',
                'model_dimension',
                'attention_heads',
                'layers_depth',
                'dropout_rate',
                'batch_size'
            ]

        print("开始全面的超参敏感性分析...")
        print("=" * 60)

        for analysis_type in analysis_types:
            if hasattr(self, f'analyze_{analysis_type}'):
                getattr(self, f'analyze_{analysis_type}')()

        # 保存结果
        self.save_results()

        # 绘制参数效率图
        self._plot_parameter_efficiency()

        return self.results

    def save_results(self):
        """保存分析结果"""
        # 保存为JSON
        with open(f'{self.analysis_dir}/results.json', 'w') as f:
            # 转换无法序列化的类型
            serializable_results = {}
            for key, value in self.results.items():
                serializable_results[key] = {str(k): v for k, v in value.items()}
            json.dump(serializable_results, f, indent=2)

        # 保存为CSV用于进一步分析
        self._save_to_csv()

        print(f"超参分析结果已保存到 {self.analysis_dir}/")

    def _save_to_csv(self):
        """将结果保存为CSV格式"""
        rows = []
        for param_type, param_results in self.results.items():
            for param_value, metrics in param_results.items():
                row = {
                    'parameter_type': param_type,
                    'parameter_value': param_value,
                    'final_accuracy': metrics['final_accuracy'],
                    'final_loss': metrics['final_loss'],
                    'convergence_epoch': metrics['convergence_epoch']
                }
                if 'parameters' in metrics:
                    row['parameters'] = metrics['parameters']
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(f'{self.analysis_dir}/results.csv', index=False)

    def _plot_learning_rate(self):
        """绘制并保存学习率分析图"""
        if 'learning_rate' not in self.results:
            return

        lr_data = self.results['learning_rate']
        lrs = list(lr_data.keys())
        accuracies = [lr_data[lr]['final_accuracy'] for lr in lrs]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogx(lrs, accuracies, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Learning Rate Sensitivity Analysis')
        ax.grid(True, alpha=0.3)

        # 标记最佳学习率
        best_idx = accuracies.index(max(accuracies))
        ax.annotate(f'Best: {lrs[best_idx]:.0e}',
                    xy=(lrs[best_idx], accuracies[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        plt.savefig(f'{self.analysis_dir}/learning_rate_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # 关闭图形以释放内存
        print(f"学习率分析图已保存: {self.analysis_dir}/learning_rate_analysis.png")

    def _plot_model_dimension(self):
        """绘制并保存模型维度分析图"""
        if 'model_dimension' not in self.results:
            return

        dim_data = self.results['model_dimension']
        dims = list(dim_data.keys())
        accuracies = [dim_data[d]['final_accuracy'] for d in dims]
        parameters = [dim_data[d]['parameters'] for d in dims]

        fig, ax = plt.subplots(figsize=(8, 6))
        # 双Y轴图
        ax2 = ax.twinx()

        line1 = ax.plot(dims, accuracies, 'o-', color='blue', linewidth=2,
                        markersize=8, label='Accuracy')
        line2 = ax2.plot(dims, parameters, 's-', color='red', linewidth=2,
                         markersize=6, label='Parameters')

        ax.set_xlabel('Model Dimension')
        ax.set_ylabel('Final Accuracy', color='blue')
        ax2.set_ylabel('Parameter Count', color='red')
        ax.set_title('Model Dimension Sensitivity Analysis')
        ax.grid(True, alpha=0.3)

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')

        plt.tight_layout()
        plt.savefig(f'{self.analysis_dir}/model_dimension_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"模型维度分析图已保存: {self.analysis_dir}/model_dimension_analysis.png")

    def _plot_attention_heads(self):
        """绘制并保存注意力头数分析图"""
        if 'attention_heads' not in self.results:
            return

        head_data = self.results['attention_heads']
        heads = list(head_data.keys())
        accuracies = [head_data[h]['final_accuracy'] for h in heads]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(heads, accuracies, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Attention Heads')
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Attention Heads Sensitivity Analysis')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.analysis_dir}/attention_heads_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"注意力头数分析图已保存: {self.analysis_dir}/attention_heads_analysis.png")

    def _plot_layers_depth(self):
        """绘制并保存网络深度分析图"""
        if 'layers_depth' not in self.results:
            return

        layer_data = self.results['layers_depth']
        layers = list(layer_data.keys())
        accuracies = [layer_data[l]['final_accuracy'] for l in layers]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(layers, accuracies, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Layers')
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Network Depth Sensitivity Analysis')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.analysis_dir}/layers_depth_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"网络深度分析图已保存: {self.analysis_dir}/layers_depth_analysis.png")

    def _plot_dropout_rate(self):
        """绘制并保存Dropout率分析图"""
        if 'dropout_rate' not in self.results:
            return

        dropout_data = self.results['dropout_rate']
        dropouts = list(dropout_data.keys())
        accuracies = [dropout_data[d]['final_accuracy'] for d in dropouts]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(dropouts, accuracies, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Dropout Rate Sensitivity Analysis')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.analysis_dir}/dropout_rate_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Dropout率分析图已保存: {self.analysis_dir}/dropout_rate_analysis.png")

    def _plot_batch_size(self):
        """绘制并保存批大小分析图"""
        if 'batch_size' not in self.results:
            return

        batch_data = self.results['batch_size']
        batches = list(batch_data.keys())
        accuracies = [batch_data[b]['final_accuracy'] for b in batches]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(batches, accuracies, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Batch Size Sensitivity Analysis')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.analysis_dir}/batch_size_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"批大小分析图已保存: {self.analysis_dir}/batch_size_analysis.png")

    def _plot_parameter_efficiency(self):
        """绘制并保存参数效率图"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 收集所有参数数量和准确率
        parameters = []
        accuracies = []
        labels = []

        for param_type in ['model_dimension', 'layers_depth']:
            if param_type in self.results:
                for param_value, metrics in self.results[param_type].items():
                    if 'parameters' in metrics:
                        parameters.append(metrics['parameters'])
                        accuracies.append(metrics['final_accuracy'])
                        labels.append(f'{param_type}_{param_value}')

        if parameters:
            ax.scatter(parameters, accuracies, s=100, alpha=0.7)
            ax.set_xlabel('Number of Parameters')
            ax.set_ylabel('Final Accuracy')
            ax.set_title('Parameter Efficiency Analysis')
            ax.grid(True, alpha=0.3)

            # 添加标签
            for i, label in enumerate(labels):
                ax.annotate(label, (parameters[i], accuracies[i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)

            plt.tight_layout()
            plt.savefig(f'{self.analysis_dir}/parameter_efficiency_analysis.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"参数效率分析图已保存: {self.analysis_dir}/parameter_efficiency_analysis.png")


def run_hyperparameter_analysis():
    """运行超参敏感性分析"""

    # 加载数据
    train_loader, val_loader, de_vocab, en_vocab = get_data_loaders()

    # 创建分析器
    analyzer = HyperparameterAnalyzer(de_vocab, en_vocab)

    # 运行分析
    results = analyzer.run_comprehensive_analysis()

    # 打印总结
    print("\n" + "=" * 60)
    print("超参敏感性分析总结")
    print("=" * 60)

    for param_type, param_results in results.items():
        best_value = max(param_results.items(),
                         key=lambda x: x[1]['final_accuracy'])
        print(f"{param_type:15} | 最佳值: {best_value[0]:8} | "
              f"准确率: {best_value[1]['final_accuracy']:.4f}")

    return results


if __name__ == "__main__":
    run_hyperparameter_analysis()