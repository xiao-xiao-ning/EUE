"""
Complete End-to-End Example
完整的端到端示例：从数据加载到实验评估
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 导入我们的模块
from old.multiview_attribution import (
    MultiViewAttributionPipeline,
    TrustScore,
    normalize_attribution
)
from deletion_experiment import (
    DeletionExperiment,
    BatchDeletionExperiment
)
from visualization import (
    AttributionVisualizer,
    UncertaintyVisualizer,
    TrustVisualizer,
    ConsistencyVisualizer
)


# ===== 1. 定义一个简单的时间序列分类模型 =====

class SimpleResNet(nn.Module):
    """简单的ResNet用于时间序列分类"""
    def __init__(self, input_channels=1, num_classes=2, length=128):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: [batch, channels, length]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        
        out = self.global_pool(out)
        out = out.squeeze(-1)
        out = self.fc(out)
        
        return out


# ===== 2. 主要实验流程 =====

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        output_dir: str = './outputs'
    ):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        
        # 初始化pipeline
        self.pipeline = MultiViewAttributionPipeline(
            model=model,
            device=device,
            wavelet='haar',
            max_level=2,
            mc_samples=30
        )
        
        # 初始化实验
        self.deletion_exp = DeletionExperiment(model, device)
        
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def run_single_sample_analysis(
        self,
        x: torch.Tensor,
        target_class: int,
        sample_name: str = 'sample'
    ):
        """
        对单个样本进行完整分析
        
        Args:
            x: [1, channels, length]
            target_class: 目标类别
            sample_name: 样本名称（用于保存文件）
        """
        print(f"\n{'='*60}")
        print(f"分析样本: {sample_name}")
        print(f"{'='*60}\n")
        
        # ===== Step 1: 计算多视图attribution =====
        print("Step 1: 计算多视图attribution和uncertainty...")
        results = self.pipeline.compute_multiview_attribution(
            x, target_class, compute_uncertainty=True
        )
        
        # 提取结果
        views = results['views']
        attributions = results['attributions']
        mapped_attributions = results['mapped_attributions']
        consistency = results['consistency']
        trust_score = results['trust_score']
        
        print(f"  ✓ 生成了 {len(views)} 个视图")
        print(f"  ✓ 跨视图一致性: {consistency:.4f}")
        
        # ===== Step 2: 计算trust-weighted attribution =====
        print("\nStep 2: 计算trust-weighted attribution...")
        mean_attr = attributions['original']['mean']
        std_attr = attributions['original']['std']
        
        trusted_attr = TrustScore.get_trusted_attribution(
            mean_attr, trust_score
        )
        
        print(f"  ✓ Trust score 范围: [{trust_score.min():.3f}, {trust_score.max():.3f}]")
        
        # ===== Step 3: 可视化 =====
        print("\nStep 3: 生成可视化...")
        
        signal = x.cpu().numpy().squeeze()
        
        # 3.1 基础可视化
        AttributionVisualizer.plot_signal_with_attribution(
            signal, mean_attr,
            title=f'{sample_name} - Attribution',
            save_path=f'{self.output_dir}/{sample_name}_basic.png'
        )
        
        # 3.2 Uncertainty可视化
        UncertaintyVisualizer.plot_mean_and_std(
            mean_attr, std_attr, signal,
            title=f'{sample_name} - Attribution with Uncertainty',
            save_path=f'{self.output_dir}/{sample_name}_uncertainty.png'
        )
        
        # 3.3 Trust可视化
        TrustVisualizer.plot_trust_comparison(
            signal, mean_attr, trust_score, trusted_attr,
            title=f'{sample_name} - Trust-aware Attribution',
            save_path=f'{self.output_dir}/{sample_name}_trust.png'
        )
        
        # 3.4 多视图对比
        AttributionVisualizer.plot_multiview_attributions(
            views, {k: attributions[k]['mean'] for k in views.keys()},
            save_path=f'{self.output_dir}/{sample_name}_multiview.png'
        )
        
        # 3.5 一致性矩阵
        ConsistencyVisualizer.plot_consistency_matrix(
            mapped_attributions,
            save_path=f'{self.output_dir}/{sample_name}_consistency.png'
        )
        
        print(f"  ✓ 可视化保存至: {self.output_dir}")
        
        # ===== Step 4: Deletion实验 =====
        print("\nStep 4: 运行Deletion实验...")
        
        # 创建对比baseline
        random_attr = np.random.randn(len(mean_attr))
        
        attributions_dict = {
            'Original Attribution': mean_attr,
            'Trust-weighted': trusted_attr,
            'Random': random_attr
        }
        
        deletion_results = self.deletion_exp.compare_attributions(
            x, target_class, attributions_dict, mode='deletion'
        )
        
        # 绘制deletion曲线
        DeletionExperiment.plot_comparison(
            deletion_results,
            mode='deletion',
            save_path=f'{self.output_dir}/{sample_name}_deletion.png'
        )
        
        # 计算AUC
        print("\n  Deletion AUC:")
        for method, (fracs, scores) in deletion_results.items():
            auc = self.deletion_exp.compute_auc(fracs, scores)
            print(f"    {method}: {auc:.4f}")
        
        print(f"\n✓ 单样本分析完成！\n")
        
        return {
            'attributions': attributions,
            'consistency': consistency,
            'trust_score': trust_score,
            'deletion_results': deletion_results
        }
    
    def run_batch_evaluation(
        self,
        dataloader: DataLoader,
        max_samples: int = 50
    ):
        """
        批量评估
        
        Args:
            dataloader: 数据加载器
            max_samples: 最大样本数
        """
        print(f"\n{'='*60}")
        print(f"批量评估 (max_samples={max_samples})")
        print(f"{'='*60}\n")
        
        # 定义attribution计算方法
        def compute_original_attribution(x, target_class):
            results = self.pipeline.compute_multiview_attribution(
                x, target_class, compute_uncertainty=True
            )
            return results['attributions']['original']['mean']
        
        def compute_trust_attribution(x, target_class):
            results = self.pipeline.compute_multiview_attribution(
                x, target_class, compute_uncertainty=True
            )
            mean_attr = results['attributions']['original']['mean']
            trust = results['trust_score']
            return TrustScore.get_trusted_attribution(mean_attr, trust)
        
        def compute_random_attribution(x, target_class):
            length = x.shape[-1]
            return np.random.randn(length)
        
        attribution_methods = {
            'Original': compute_original_attribution,
            'Trust-weighted': compute_trust_attribution,
            'Random': compute_random_attribution
        }
        
        # 运行批量实验
        batch_exp = BatchDeletionExperiment(self.model, self.device)
        
        auc_scores = batch_exp.run_on_dataset(
            dataloader,
            attribution_methods,
            mode='deletion',
            max_samples=max_samples
        )
        
        # 总结结果
        summary = BatchDeletionExperiment.summarize_results(auc_scores)
        
        print("\n批量Deletion实验结果:")
        print("-" * 60)
        for method, stats in summary.items():
            print(f"\n{method}:")
            print(f"  Mean AUC:   {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Median AUC: {stats['median']:.4f}")
            print(f"  Range:      [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # 可视化对比
        plt.figure(figsize=(10, 6))
        methods = list(summary.keys())
        means = [summary[m]['mean'] for m in methods]
        stds = [summary[m]['std'] for m in methods]
        
        x_pos = np.arange(len(methods))
        plt.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5)
        plt.xticks(x_pos, methods)
        plt.ylabel('Mean Deletion AUC', fontsize=12)
        plt.title('Batch Evaluation: Deletion AUC Comparison', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/batch_evaluation.png', dpi=300)

        print(f"\n✓ 批量评估完成！")
        
        return summary


# ===== 3. 使用示例 =====

def main():
    """主函数 - 演示完整流程"""
    
    print("\n" + "="*60)
    print("Multi-view Trust-aware Attribution - 完整示例")
    print("="*60)
    
    # ===== 设置 =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # ===== 创建模拟数据 =====
    print("\n创建模拟数据...")
    
    # 简单的合成数据：两类时间序列
    def create_synthetic_data(n_samples=100, length=128):
        X = []
        y = []
        
        for i in range(n_samples):
            t = np.linspace(0, 10, length)
            
            if i < n_samples // 2:
                # Class 0: 低频正弦
                signal = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(length)
                label = 0
            else:
                # Class 1: 高频正弦
                signal = np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(length)
                label = 1
            
            X.append(signal)
            y.append(label)
        
        X = np.array(X)[:, np.newaxis, :]  # [n_samples, 1, length]
        y = np.array(y)
        
        return torch.FloatTensor(X), torch.LongTensor(y)
    
    X_train, y_train = create_synthetic_data(n_samples=100, length=128)
    X_test, y_test = create_synthetic_data(n_samples=20, length=128)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    # ===== 创建并训练模型 =====
    print("\n创建并训练模型...")
    
    model = SimpleResNet(input_channels=1, num_classes=2, length=128)
    model = model.to(device)
    
    # 简单训练（实际使用时应该更复杂）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}")
    
    model.eval()
    print("  ✓ 模型训练完成")
    
    # ===== 运行实验 =====
    runner = ExperimentRunner(
        model=model,
        device=device,
        output_dir='./experiment_outputs'
    )
    
    # 单样本分析
    test_x, test_y = X_test[0:1], y_test[0]
    
    single_results = runner.run_single_sample_analysis(
        test_x,
        test_y.item(),
        sample_name='test_sample_0'
    )
    
    # 批量评估
    batch_results = runner.run_batch_evaluation(
        test_loader,
        max_samples=20
    )
    
    print("\n" + "="*60)
    print("实验完成！所有结果已保存至 ./experiment_outputs/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()