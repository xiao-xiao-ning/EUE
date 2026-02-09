"""
Unified Experiment Pipeline
整合版完整实验流程

包含所有必要的实验：
1. Uncertainty Analysis（解释不确定性分析）
2. Time-step Consistency（时间步级别一致性）
3. Trust vs Uncertainty（区分可信度和不确定性）
4. Deletion Experiment（验证有效性）
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import seaborn as sns

from unified_framework import UnifiedMultiViewPipeline
from core_trust_definition import ThreeConceptsDistinction


class UnifiedExperimentPipeline:
    """
    整合版实验Pipeline
    """
    
    def __init__(
        self,
        model,
        device='cuda',
        output_dir='./unified_results'
    ):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化pipeline
        self.pipeline = UnifiedMultiViewPipeline(
            model,
            device=device,
            mc_samples=20,
            trust_epsilon=0.1
        )
    
    def exp1_uncertainty_analysis(self, x, y, name='sample'):
        """
        实验1：Uncertainty Analysis
        展示uncertainty反映attribution的不稳定性
        """
        print(f"\n{'='*70}")
        print("实验1：Explanation Uncertainty Analysis")
        print(f"{'='*70}")
        
        results = self.pipeline.compute_complete_explanation(
            x, y, compute_trust=False  # 先不计算trust，只看uncertainty
        )
        
        signal = x.cpu().numpy().squeeze()
        if signal.ndim == 2:
            signal = signal[0]
        
        mean_attr = results['attribution_mean']
        std_attr = results['attribution_std']
        
        # 可视化
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        
        # 1. 信号
        axes[0].plot(signal, linewidth=1.5, color='steelblue')
        axes[0].set_ylabel('Signal', fontsize=11)
        axes[0].set_title(f'{name} - Uncertainty Analysis', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Attribution with uncertainty
        axes[1].plot(mean_attr, linewidth=2, color='darkred', label='Mean')
        axes[1].fill_between(
            range(len(mean_attr)),
            mean_attr - std_attr,
            mean_attr + std_attr,
            alpha=0.3, color='coral', label='±1 Std'
        )
        axes[1].set_ylabel('Attribution', fontsize=11)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Uncertainty (Std)
        axes[2].plot(std_attr, linewidth=2, color='orange')
        axes[2].fill_between(range(len(std_attr)), 0, std_attr, 
                            alpha=0.3, color='orange')
        axes[2].set_xlabel('Time Step', fontsize=11)
        axes[2].set_ylabel('Uncertainty (Std)', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{name}_exp1_uncertainty.png', dpi=300)
        plt.close()
        
        print(f"✓ 保存: {name}_exp1_uncertainty.png")
        print(f"  Mean uncertainty: {std_attr.mean():.4f}")
        
        return results
    
    def exp2_timestep_consistency(self, x, y, name='sample'):
        """
        实验2：Time-step-level Consistency
        展示每个时间步的跨视图一致性
        """
        print(f"\n{'='*70}")
        print("实验2：Time-step-level Cross-view Consistency")
        print(f"{'='*70}")
        
        results = self.pipeline.compute_complete_explanation(
            x, y, compute_trust=False
        )
        
        signal = x.cpu().numpy().squeeze()
        if signal.ndim == 2:
            signal = signal[0]
        
        mean_attr = results['attribution_mean']
        consistency = results['consistency']
        
        # 可视化
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        
        # 1. 信号
        axes[0].plot(signal, linewidth=1.5, color='steelblue')
        axes[0].set_ylabel('Signal', fontsize=11)
        axes[0].set_title(f'{name} - Timestep-level Consistency', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Attribution
        axes[1].bar(range(len(mean_attr)), mean_attr, alpha=0.7, color='coral')
        axes[1].set_ylabel('Attribution', fontsize=11)
        axes[1].axhline(y=0, color='gray', linestyle='--')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Consistency
        axes[2].plot(consistency, linewidth=2, color='green')
        axes[2].fill_between(range(len(consistency)), 0, consistency,
                            alpha=0.3, color='green')
        axes[2].set_xlabel('Time Step', fontsize=11)
        axes[2].set_ylabel('Consistency', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{name}_exp2_consistency.png', dpi=300)
        plt.close()
        
        print(f"✓ 保存: {name}_exp2_consistency.png")
        print(f"  Mean consistency: {consistency.mean():.4f}")
        
        return results
    
    def exp3_trust_vs_uncertainty(self, x, y, name='sample'):
        """
        实验3：Trust vs Uncertainty
        关键！证明Trust ≠ Uncertainty
        
        使用Trust_agg公式：
        Trust_agg(t) = (1/R) Σ_r exp(-U_r(t)) · C_r(t) · A_r(t)
        
        这个公式整合了：
        - 多视图信息（R个视图）
        - 不确定性（exp(-U)：低不确定性→高权重）
        - 一致性（C_t：跨视图一致性）
        - 归因值（A_r(t)：重要性）
        """
        print(f"\n{'='*70}")
        print("实验3：Trust vs Uncertainty - 核心区别")
        print(f"{'='*70}")
        
        print("  计算Trust_agg（整合多视图信息）...")
        results = self.pipeline.compute_complete_explanation(
            x, y, 
            compute_trust=True,
            trust_method='aggregated'  # 使用Trust_agg
        )
        
        signal = x.cpu().numpy().squeeze()
        if signal.ndim == 2:
            signal = signal[0]
        
        mean_attr = results['attribution_mean']
        uncertainty = results['attribution_std']
        trust = results['trust']
        categories = results['categories']
        
        # 统计
        print("\n  时间步分类:")
        key_categories = [
            'stable_but_untrusted',
            'unstable_but_trusted',
            'stable_and_trusted',
            'unstable_and_untrusted'
        ]
        for cat in key_categories:
            if cat in categories:
                print(f"    {cat}: {len(categories[cat])} 个")
        
        # 可视化
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        # 1. Attribution
        axes[0].bar(range(len(mean_attr)), mean_attr, alpha=0.7, color='coral')
        axes[0].set_ylabel('Attribution\n(Importance)', fontsize=11)
        axes[0].set_title(f'{name} - Trust vs Uncertainty', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Uncertainty
        unc_norm = uncertainty / (uncertainty.max() + 1e-8)
        axes[1].plot(unc_norm, linewidth=2, color='orange')
        axes[1].fill_between(range(len(unc_norm)), 0, unc_norm,
                            alpha=0.3, color='orange')
        axes[1].set_ylabel('Uncertainty\n(Stability)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # 3. Trust
        axes[2].plot(trust, linewidth=2, color='darkgreen')
        axes[2].fill_between(range(len(trust)), 0, trust,
                            alpha=0.3, color='green')
        axes[2].set_ylabel('Trust\n(Verifiability)', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        # 4. 分类展示
        colors = {
            'stable_but_untrusted': 'red',
            'unstable_but_trusted': 'blue',
            'stable_and_trusted': 'darkgreen',
            'unstable_and_untrusted': 'gray'
        }
        
        for cat_name in key_categories:
            if cat_name in categories and len(categories[cat_name]) > 0:
                indices = categories[cat_name]
                axes[3].scatter(
                    indices, np.ones(len(indices)),
                    c=colors.get(cat_name, 'gray'),
                    s=100, alpha=0.7,
                    label=cat_name.replace('_', ' ').title()
                )
        
        axes[3].set_xlabel('Time Step', fontsize=11)
        axes[3].set_ylabel('Category', fontsize=11)
        axes[3].set_ylim([0.5, 1.5])
        axes[3].set_yticks([])
        axes[3].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        axes[3].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{name}_exp3_trust_vs_uncertainty.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存: {name}_exp3_trust_vs_uncertainty.png")
        
        return results
    
    def exp4_deletion(self, x, y, name='sample'):
        """
        实验4：Deletion Experiment
        验证trust-weighted attribution的有效性
        
        使用Trust_agg进行加权
        """
        print(f"\n{'='*70}")
        print("实验4：Deletion Experiment")
        print(f"{'='*70}")
        
        results = self.pipeline.compute_complete_explanation(
            x, y, 
            compute_trust=True, 
            trust_method='aggregated'  # 使用Trust_agg
        )
        
        mean_attr = results['attribution_mean']
        trust = results['trust']
        trusted_attr = results.get('trusted_importance', mean_attr * trust)
        
        # Random baseline
        random_attr = np.random.randn(len(mean_attr))
        
        # Deletion曲线
        from deletion_experiment import DeletionExperiment
        
        deletion_exp = DeletionExperiment(self.model, self.device)
        
        attributions = {
            'Original': mean_attr,
            'Trust-weighted': trusted_attr,
            'Random': random_attr
        }
        
        deletion_results = deletion_exp.compare_attributions(
            x, y, attributions, mode='deletion'
        )
        
        # 计算AUC
        print("\n  Deletion AUC:")
        for method, (fracs, scores) in deletion_results.items():
            auc = deletion_exp.compute_auc(fracs, scores)
            print(f"    {method}: {auc:.4f}")
        
        # 可视化
        DeletionExperiment.plot_comparison(
            deletion_results,
            mode='deletion',
            save_path=str(self.output_dir / f'{name}_exp4_deletion.png')
        )
        
        print(f"✓ 保存: {name}_exp4_deletion.png")
        
        return results
    
    def run_all_experiments(self, x, y, name='sample'):
        """运行所有实验"""
        print("\n" + "="*70)
        print(f"运行完整实验: {name}")
        print("="*70)
        
        self.exp1_uncertainty_analysis(x, y, name)
        self.exp2_timestep_consistency(x, y, name)
        self.exp3_trust_vs_uncertainty(x, y, name)
        self.exp4_deletion(x, y, name)
        
        print("\n" + "="*70)
        print("所有实验完成！")
        print(f"结果保存在: {self.output_dir}")
        print("="*70)


if __name__ == "__main__":
    print("整合版实验Pipeline已加载")
    print("\n使用方法:")
    print("""
    from unified_experiments import UnifiedExperimentPipeline
    
    # 创建pipeline
    exp = UnifiedExperimentPipeline(model, device='cuda')
    
    # 运行所有实验
    exp.run_all_experiments(x, y, name='sample_1')
    """)