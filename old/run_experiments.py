"""
Run Experiments - 完整实验运行脚本
一键运行所有实验：训练、评估、可视化
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json

# 导入我们的模块
from data_loader import UCRDataLoader, CustomDataLoader
from model_loader import SimpleResNet, FCN, SimpleTrainer, save_model, load_model
from old.multiview_attribution import MultiViewAttributionPipeline, TrustScore
from deletion_experiment import DeletionExperiment, BatchDeletionExperiment
from visualization import (
    AttributionVisualizer,
    UncertaintyVisualizer,
    TrustVisualizer,
    ConsistencyVisualizer
)


class ExperimentRunner:
    """
    完整的实验运行器
    """
    def __init__(
        self,
        dataset_name: str = 'ArrowHead',
        model_type: str = 'SimpleResNet',
        device: str = 'cuda',
        output_dir: str = './outputs',
        ucr_path: str = './data/raw/UCR'
    ):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path(output_dir)
        self.ucr_path = ucr_path
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'experiments').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        
        print(f"实验配置:")
        print(f"  数据集: {dataset_name}")
        print(f"  模型: {model_type}")
        print(f"  设备: {self.device}")
        print(f"  输出目录: {output_dir}")
    
    def step1_load_data(self):
        """步骤1：加载数据"""
        print("\n" + "="*60)
        print("步骤1：加载数据")
        print("="*60)
        
        loader = UCRDataLoader(self.ucr_path)
        
        # 列出可用数据集
        available = loader.list_available_datasets()
        if not available:
            print("❌ 未找到UCR数据集!")
            print("请从 https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ 下载")
            return None, None
        
        print(f"找到 {len(available)} 个UCR数据集")
        
        # 检查目标数据集是否存在
        if self.dataset_name not in available:
            print(f"❌ 数据集 {self.dataset_name} 不存在")
            print(f"可用数据集: {', '.join(available[:10])}...")
            return None, None
        
        # 加载数据
        train_loader, test_loader = loader.create_dataloaders(
            self.dataset_name,
            batch_size=32
        )
        
        # 获取数据信息
        sample_x, sample_y = next(iter(train_loader))
        n_channels = sample_x.shape[1]
        length = sample_x.shape[2]
        n_classes = len(torch.unique(sample_y))
        
        self.data_info = {
            'n_channels': n_channels,
            'length': length,
            'n_classes': n_classes
        }
        
        print(f"✓ 数据加载成功")
        print(f"  输入shape: [batch, {n_channels}, {length}]")
        print(f"  类别数: {n_classes}")
        
        return train_loader, test_loader
    
    def step2_train_or_load_model(self, train_loader, test_loader):
        """步骤2：训练或加载模型"""
        print("\n" + "="*60)
        print("步骤2：准备模型")
        print("="*60)
        
        model_path = self.output_dir / 'models' / f'{self.dataset_name}_{self.model_type}.pth'
        
        # 检查是否已有训练好的模型
        if model_path.exists():
            print(f"发现已训练模型: {model_path}")
            response = input("是否加载已有模型？(y/n): ")
            
            if response.lower() == 'y':
                if self.model_type == 'SimpleResNet':
                    model = load_model(SimpleResNet, str(model_path), self.device)
                elif self.model_type == 'FCN':
                    model = load_model(FCN, str(model_path), self.device)
                else:
                    model = load_model(SimpleResNet, str(model_path), self.device)
                
                print("✓ 模型加载成功")
                return model
        
        # 训练新模型
        print("训练新模型...")
        
        if self.model_type == 'SimpleResNet':
            model = SimpleResNet(
                input_channels=self.data_info['n_channels'],
                num_classes=self.data_info['n_classes'],
                length=self.data_info['length']
            )
        elif self.model_type == 'FCN':
            model = FCN(
                input_channels=self.data_info['n_channels'],
                num_classes=self.data_info['n_classes'],
                length=self.data_info['length']
            )
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")
        
        # 训练
        trainer = SimpleTrainer(model, device=self.device, lr=0.001)
        best_acc = trainer.train(
            train_loader,
            test_loader,
            epochs=50,
            save_path=str(model_path),
            verbose=True
        )
        
        print(f"✓ 模型训练完成，最佳准确率: {best_acc:.2f}%")
        
        return model
    
    def step3_run_single_sample_analysis(self, model, test_loader):
        """步骤3：单样本分析"""
        print("\n" + "="*60)
        print("步骤3：单样本分析")
        print("="*60)
        
        # 选择几个测试样本
        n_samples = min(5, len(test_loader.dataset))
        
        results = []
        
        for idx in range(n_samples):
            x, y = test_loader.dataset[idx]
            x = x.unsqueeze(0)  # [1, channels, length]
            
            print(f"\n分析样本 {idx+1}/{n_samples} (真实标签: {y})")
            
            # 创建pipeline
            pipeline = MultiViewAttributionPipeline(
                model=model,
                device=self.device,
                mc_samples=20
            )
            
            # 计算multi-view attribution
            result = pipeline.compute_multiview_attribution(
                x, y, compute_uncertainty=True
            )
            
            print(f"  一致性: {result['consistency']:.4f}")
            print(f"  Trust score: {result['trust_score'].mean():.4f}")
            
            # 提取结果
            mean_attr = result['attributions']['original']['mean']
            std_attr = result['attributions']['original']['std']
            trust_score = result['trust_score']
            trusted_attr = TrustScore.get_trusted_attribution(mean_attr, trust_score)
            
            # 可视化
            signal = x.cpu().numpy().squeeze()
            
            # 1. Uncertainty可视化
            UncertaintyVisualizer.plot_mean_and_std(
                mean_attr, std_attr, signal,
                title=f'{self.dataset_name} Sample {idx} - Uncertainty',
                save_path=str(self.output_dir / 'visualizations' / f'sample_{idx}_uncertainty.png')
            )
            
            # 2. Trust可视化
            TrustVisualizer.plot_trust_comparison(
                signal, mean_attr, trust_score, trusted_attr,
                title=f'{self.dataset_name} Sample {idx} - Trust',
                save_path=str(self.output_dir / 'visualizations' / f'sample_{idx}_trust.png')
            )
            
            # 3. 一致性矩阵
            ConsistencyVisualizer.plot_consistency_matrix(
                result['mapped_attributions'],
                save_path=str(self.output_dir / 'visualizations' / f'sample_{idx}_consistency.png')
            )
            
            results.append({
                'idx': idx,
                'label': int(y),
                'consistency': float(result['consistency']),
                'mean_trust': float(trust_score.mean())
            })
        
        # 保存结果
        with open(self.output_dir / 'experiments' / 'single_sample_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ 单样本分析完成，结果保存至: {self.output_dir / 'visualizations'}")
        
        return results
    
    def step4_run_deletion_experiments(self, model, test_loader):
        """步骤4：Deletion实验"""
        print("\n" + "="*60)
        print("步骤4：Deletion实验")
        print("="*60)
        
        # 批量评估
        batch_exp = BatchDeletionExperiment(model, self.device)
        
        # 定义attribution方法
        def compute_original(x, y):
            pipeline = MultiViewAttributionPipeline(
                model, self.device, mc_samples=10  # 用更少采样加快速度
            )
            result = pipeline.compute_multiview_attribution(x, y, compute_uncertainty=True)
            return result['attributions']['original']['mean']
        
        def compute_trust(x, y):
            pipeline = MultiViewAttributionPipeline(
                model, self.device, mc_samples=10
            )
            result = pipeline.compute_multiview_attribution(x, y, compute_uncertainty=True)
            mean_attr = result['attributions']['original']['mean']
            trust = result['trust_score']
            return TrustScore.get_trusted_attribution(mean_attr, trust)
        
        def compute_random(x, y):
            length = x.shape[-1]
            return np.random.randn(length)
        
        attribution_methods = {
            'Original': compute_original,
            'Trust-weighted': compute_trust,
            'Random': compute_random
        }
        
        # 运行实验
        max_samples = min(30, len(test_loader.dataset))
        print(f"在 {max_samples} 个样本上运行deletion实验...")
        
        auc_scores = batch_exp.run_on_dataset(
            test_loader,
            attribution_methods,
            mode='deletion',
            max_samples=max_samples
        )
        
        # 总结结果
        summary = BatchDeletionExperiment.summarize_results(auc_scores)
        
        print("\n实验结果:")
        print("-" * 60)
        for method, stats in summary.items():
            print(f"{method}:")
            print(f"  Mean AUC: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # 可视化
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        methods = list(summary.keys())
        means = [summary[m]['mean'] for m in methods]
        stds = [summary[m]['std'] for m in methods]
        
        x_pos = np.arange(len(methods))
        plt.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5,
               color=['coral', 'darkgreen', 'gray'])
        plt.xticks(x_pos, methods)
        plt.ylabel('Mean Deletion AUC', fontsize=12)
        plt.title(f'{self.dataset_name} - Deletion Experiment Results', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / 'experiments' / 'deletion_results.png',
            dpi=300, bbox_inches='tight'
        )
        
        # 保存结果
        with open(self.output_dir / 'experiments' / 'deletion_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Deletion实验完成，结果保存至: {self.output_dir / 'experiments'}")
        
        return summary
    
    def run_all(self):
        """运行所有实验"""
        print("\n" + "="*60)
        print(f"开始完整实验流程: {self.dataset_name}")
        print("="*60)
        
        # 步骤1：加载数据
        train_loader, test_loader = self.step1_load_data()
        if train_loader is None:
            return
        
        # 步骤2：训练/加载模型
        model = self.step2_train_or_load_model(train_loader, test_loader)
        
        # 步骤3：单样本分析
        single_results = self.step3_run_single_sample_analysis(model, test_loader)
        
        # 步骤4：Deletion实验
        deletion_results = self.step4_run_deletion_experiments(model, test_loader)
        
        print("\n" + "="*60)
        print("所有实验完成！")
        print("="*60)
        print(f"\n结果保存在: {self.output_dir}")
        print(f"  - 可视化: {self.output_dir / 'visualizations'}")
        print(f"  - 实验结果: {self.output_dir / 'experiments'}")
        print(f"  - 模型: {self.output_dir / 'models'}")


# ===== 命令行接口 =====

def main():
    parser = argparse.ArgumentParser(description='运行Multi-view Attribution实验')
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='ArrowHead',
        help='UCR数据集名称（如ArrowHead, GunPoint）'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='SimpleResNet',
        choices=['SimpleResNet', 'FCN'],
        help='模型类型'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='计算设备（cuda或cpu）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./outputs',
        help='输出目录'
    )
    
    parser.add_argument(
        '--ucr-path',
        type=str,
        default='./data/raw/UCR',
        help='UCR数据集路径'
    )
    
    args = parser.parse_args()
    
    # 运行实验
    runner = ExperimentRunner(
        dataset_name=args.dataset,
        model_type=args.model,
        device=args.device,
        output_dir=args.output,
        ucr_path=args.ucr_path
    )
    
    runner.run_all()


if __name__ == "__main__":
    # 如果直接运行（不使用命令行参数）
    import sys
    
    if len(sys.argv) == 1:
        # 交互式模式
        print("="*60)
        print("Multi-view Trust-aware Attribution - 实验运行器")
        print("="*60)
        print("\n选择运行模式:")
        print("  1. 交互式运行（推荐）")
        print("  2. 使用默认配置运行")
        print("  3. 退出")
        
        choice = input("\n请选择 (1/2/3): ")
        
        if choice == '1':
            # 交互式配置
            print("\n请输入配置:")
            dataset = input("数据集名称 (默认: ArrowHead): ") or 'ArrowHead'
            model = input("模型类型 (SimpleResNet/FCN, 默认: SimpleResNet): ") or 'SimpleResNet'
            
            runner = ExperimentRunner(
                dataset_name=dataset,
                model_type=model
            )
            runner.run_all()
        
        elif choice == '2':
            # 默认配置
            runner = ExperimentRunner()
            runner.run_all()
        
        else:
            print("退出")
    
    else:
        # 命令行模式
        main()