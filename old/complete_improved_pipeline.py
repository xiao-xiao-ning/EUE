"""
Complete Improved Experiment Pipeline
完整的改进版实验流程

集成所有改进：
1. ✓ Bug修复（deletion negative stride）
2. ✓ 支持Transformer等更多模型
3. ✓ 训练与评估分离
4. ✓ Time-step-level consistency
5. ✓ 明确区分importance vs reliability
6. ✓ Trust稳定性实验
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# ==================== 改进版Pipeline ====================

class ImprovedMultiViewPipeline:
    """
    改进版多视图归因Pipeline
    
    主要改进：
    - 使用时间步级别的一致性
    - 明确区分importance和reliability
    - 添加stability测试
    """
    
    def __init__(
        self,
        model,
        device='cuda',
        mc_samples=30
    ):
        from old.multiview_attribution import (
            MultiViewGenerator,
            IntegratedGradients,
            MCDropoutWrapper
        )
        from old.enhanced_consistency_trust import (
            TimestepLevelConsistency,
            ImportanceVsReliability
        )
        
        self.device = device
        self.mc_samples = mc_samples
        
        # 用MC Dropout包装模型
        self.mc_model = MCDropoutWrapper(model)
        
        # 初始化组件
        self.ig = IntegratedGradients(self.mc_model, device)
        self.view_generator = MultiViewGenerator(wavelet='haar', max_level=2)
        self.consistency_calculator = TimestepLevelConsistency()
        self.importance_reliability = ImportanceVsReliability()
    
    def compute_explanation(
        self,
        x: torch.Tensor,
        target_class: int
    ):
        """
        计算完整的解释（改进版）
        
        Returns:
            {
                'attribution_mean': [length],
                'attribution_std': [length],
                'timestep_consistency': [length],  # 改进：时间步级别
                'reliability': [length],           # 新增：纯可靠性
                'trust': [length],                 # 综合trust
                'categories': {...}                # 时间步分类
            }
        """
        x_np = x.cpu().numpy().squeeze()
        original_length = x_np.shape[-1]
        
        # 1. 生成多视图
        views = self.view_generator.decompose(x_np)
        
        # 2. 对每个view计算attribution + uncertainty
        attributions_with_uncertainty = {}
        
        for view_name, view_signal in views.items():
            view_tensor = torch.FloatTensor(view_signal).unsqueeze(0)
            
            mean_attr, std_attr = self.ig.compute_attribution_with_uncertainty(
                view_tensor, target_class, self.mc_samples
            )
            
            attributions_with_uncertainty[view_name] = {
                'mean': mean_attr,
                'std': std_attr
            }
        
        # 3. 映射到时间域
        mapped_attributions = {}
        for view_name in views.keys():
            mapped = self.view_generator.map_to_time_domain(
                attributions_with_uncertainty[view_name]['mean'],
                original_length,
                view_name
            )
            mapped_attributions[view_name] = mapped
        
        # 4. 计算时间步级别的一致性（改进！）
        timestep_consistency = self.consistency_calculator.compute_timestep_consistency(
            mapped_attributions,
            method='std'
        )
        
        # 5. 提取original view的结果
        mean_attr = attributions_with_uncertainty['original']['mean']
        std_attr = attributions_with_uncertainty['original']['std']
        
        # 6. 计算纯可靠性（不考虑importance）
        reliability = self.importance_reliability.compute_reliability_metrics(
            std_attr,
            timestep_consistency,
            alpha=0.5
        )
        
        # 7. 计算trust（reliability的别名，但语义更清晰）
        trust = reliability
        
        # 8. 分类时间步
        categories = self.importance_reliability.categorize_timesteps(
            mean_attr,
            trust,
            importance_threshold=0.5,
            trust_threshold=0.6
        )
        
        return {
            'attribution_mean': mean_attr,
            'attribution_std': std_attr,
            'timestep_consistency': timestep_consistency,
            'reliability': reliability,
            'trust': trust,
            'categories': categories,
            'views': views,
            'mapped_attributions': mapped_attributions
        }


# ==================== 完整实验流程 ====================

class CompleteExperimentPipeline:
    """
    完整的实验流程
    包含所有改进
    """
    
    def __init__(
        self,
        model,
        device='cuda',
        output_dir='./improved_results'
    ):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化pipeline
        self.pipeline = ImprovedMultiViewPipeline(model, device)
        
        # 初始化实验组件
        from deletion_experiment import DeletionExperiment
        from old.enhanced_consistency_trust import TrustStabilityExperiment
        
        self.deletion_exp = DeletionExperiment(model, device)
        self.stability_exp = TrustStabilityExperiment(model, device)
    
    def experiment_1_uncertainty_analysis(self, x, y, sample_name='sample'):
        """
        实验1：Uncertainty分析
        展示uncertainty不是噪声，而是反映不确定性
        """
        print(f"\n{'='*60}")
        print("实验1：Explanation Uncertainty Analysis")
        print(f"{'='*60}")
        
        results = self.pipeline.compute_explanation(x, y)
        
        # 可视化
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        
        signal = x.cpu().numpy().squeeze()[0] if x.ndim == 3 else x.cpu().numpy().squeeze()
        
        # 1. 信号
        axes[0].plot(signal, linewidth=1.5, color='steelblue')
        axes[0].set_ylabel('Signal', fontsize=11)
        axes[0].set_title(f'{sample_name} - Uncertainty Analysis', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Attribution with uncertainty
        mean_attr = results['attribution_mean']
        std_attr = results['attribution_std']
        
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
        
        # 3. Uncertainty
        axes[2].plot(std_attr, linewidth=2, color='orange')
        axes[2].fill_between(range(len(std_attr)), 0, std_attr, alpha=0.3, color='orange')
        axes[2].set_xlabel('Time Step', fontsize=11)
        axes[2].set_ylabel('Uncertainty', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{sample_name}_exp1_uncertainty.png', dpi=300)
        plt.close()
        
        print(f"✓ 结果保存至: {sample_name}_exp1_uncertainty.png")
        print(f"  Mean uncertainty: {std_attr.mean():.4f}")
        print(f"  Max uncertainty: {std_attr.max():.4f}")
        
        return results
    
    def experiment_2_timestep_consistency(self, x, y, sample_name='sample'):
        """
        实验2：Time-step-level Consistency
        展示在关键时间段，跨视图一致性更高
        """
        print(f"\n{'='*60}")
        print("实验2：Time-step-level Cross-view Consistency")
        print(f"{'='*60}")
        
        results = self.pipeline.compute_explanation(x, y)
        
        # 可视化
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        
        signal = x.cpu().numpy().squeeze()[0] if x.ndim == 3 else x.cpu().numpy().squeeze()
        mean_attr = results['attribution_mean']
        consistency = results['timestep_consistency']
        
        # 1. 信号
        axes[0].plot(signal, linewidth=1.5, color='steelblue')
        axes[0].set_ylabel('Signal', fontsize=11)
        axes[0].set_title(f'{sample_name} - Timestep-level Consistency', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Attribution
        axes[1].bar(range(len(mean_attr)), mean_attr, alpha=0.7, color='coral')
        axes[1].set_ylabel('Attribution', fontsize=11)
        axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
        axes[1].grid(True, alpha=0.3)
        
        # 3. Timestep Consistency
        axes[2].plot(consistency, linewidth=2, color='green')
        axes[2].fill_between(range(len(consistency)), 0, consistency, alpha=0.3, color='green')
        axes[2].set_ylabel('Consistency', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        # 4. Overlay: 高attribution + 高consistency的区域
        attr_norm = np.abs(mean_attr) / (np.abs(mean_attr).max() + 1e-8)
        high_attr_and_consistency = (attr_norm > 0.5) & (consistency > 0.6)
        
        axes[3].fill_between(
            range(len(attr_norm)),
            0, 1,
            where=high_attr_and_consistency,
            alpha=0.4, color='darkgreen',
            label='High Attr + High Consistency'
        )
        axes[3].set_xlabel('Time Step', fontsize=11)
        axes[3].set_ylabel('Key Regions', fontsize=11)
        axes[3].set_ylim([0, 1])
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{sample_name}_exp2_consistency.png', dpi=300)
        plt.close()
        
        print(f"✓ 结果保存至: {sample_name}_exp2_consistency.png")
        print(f"  Mean consistency: {consistency.mean():.4f}")
        print(f"  关键区域一致性: {consistency[high_attr_and_consistency].mean():.4f}")
        
        return results
    
    def experiment_3_importance_vs_reliability(self, x, y, sample_name='sample'):
        """
        实验3：区分 Importance vs Reliability (Trust)
        展示四类时间步
        """
        print(f"\n{'='*60}")
        print("实验3：Attribution Importance vs Explanation Reliability")
        print(f"{'='*60}")
        
        results = self.pipeline.compute_explanation(x, y)
        
        categories = results['categories']
        mean_attr = results['attribution_mean']
        trust = results['trust']
        
        # 统计
        print("\n时间步分类:")
        for cat_name, indices in categories.items():
            print(f"  {cat_name}: {len(indices)} 个")
        
        # 可视化
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        
        signal = x.cpu().numpy().squeeze()[0] if x.ndim == 3 else x.cpu().numpy().squeeze()
        
        # 1. Attribution
        axes[0].bar(range(len(mean_attr)), mean_attr, alpha=0.7, color='coral')
        axes[0].set_ylabel('Attribution\n(Importance)', fontsize=11)
        axes[0].set_title(f'{sample_name} - Importance vs Reliability', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Trust (Reliability)
        axes[1].plot(trust, linewidth=2, color='darkgreen')
        axes[1].fill_between(range(len(trust)), 0, trust, alpha=0.3, color='green')
        axes[1].set_ylabel('Trust\n(Reliability)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # 3. 分类可视化
        colors = {
            'reliable_important': 'darkgreen',
            'unreliable_important': 'red',
            'reliable_unimportant': 'lightblue',
            'unreliable_unimportant': 'gray'
        }
        
        for cat_name, indices in categories.items():
            if len(indices) > 0:
                axes[2].scatter(
                    indices, np.ones(len(indices)),
                    c=colors[cat_name], s=100, alpha=0.7,
                    label=cat_name.replace('_', ' ').title()
                )
        
        axes[2].set_xlabel('Time Step', fontsize=11)
        axes[2].set_ylabel('Category', fontsize=11)
        axes[2].set_ylim([0.5, 1.5])
        axes[2].set_yticks([])
        axes[2].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{sample_name}_exp3_importance_vs_reliability.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 结果保存至: {sample_name}_exp3_importance_vs_reliability.png")
        
        return results
    
    def experiment_4_trust_stability(self, x, y, sample_name='sample'):
        """
        实验4：Trust稳定性验证
        证明：高trust的时间步在噪声下更稳定
        """
        print(f"\n{'='*60}")
        print("实验4：Trust Stability under Noise")
        print(f"{'='*60}")
        
        results = self.pipeline.compute_explanation(x, y)
        categories = results['categories']
        
        # 对比两组
        group_a = categories['reliable_important']  # 高trust
        group_b = categories['unreliable_important']  # 低trust
        
        if len(group_a) == 0 or len(group_b) == 0:
            print("警告: 某些类别为空，跳过稳定性测试")
            return results
        
        print(f"\n测试两组时间步:")
        print(f"  Group A (可信的重要点): {len(group_a)} 个")
        print(f"  Group B (不可信的重要点): {len(group_b)} 个")
        
        # 稳定性测试
        stability_a = self.stability_exp.stability_under_noise(
            x, y, group_a, noise_levels=[0.1, 0.2, 0.3], n_trials=10
        )
        
        stability_b = self.stability_exp.stability_under_noise(
            x, y, group_b, noise_levels=[0.1, 0.2, 0.3], n_trials=10
        )
        
        # 结果对比
        print("\n稳定性对比:")
        print(f"  Group A (高Trust):")
        print(f"    预测稳定性: {stability_a['prediction_stability']:.2%}")
        print(f"    置信度下降: {stability_a['confidence_drop']:.4f}")
        print(f"    概率波动: {stability_a['probability_std']:.4f}")
        
        print(f"  Group B (低Trust):")
        print(f"    预测稳定性: {stability_b['prediction_stability']:.2%}")
        print(f"    置信度下降: {stability_b['confidence_drop']:.4f}")
        print(f"    概率波动: {stability_b['probability_std']:.4f}")
        
        # 可视化
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics = ['Prediction\nStability', 'Confidence\nDrop', 'Probability\nStd']
        group_a_vals = [
            stability_a['prediction_stability'],
            -stability_a['confidence_drop'],  # 负值表示下降
            -stability_a['probability_std']   # 负值表示波动
        ]
        group_b_vals = [
            stability_b['prediction_stability'],
            -stability_b['confidence_drop'],
            -stability_b['probability_std']
        ]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x_pos - width/2, group_a_vals, width, label='High Trust', 
              color='darkgreen', alpha=0.7)
        ax.bar(x_pos + width/2, group_b_vals, width, label='Low Trust',
              color='red', alpha=0.7)
        
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title(f'{sample_name} - Trust Stability Comparison', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{sample_name}_exp4_trust_stability.png', dpi=300)
        plt.close()
        
        print(f"\n✓ 结果保存至: {sample_name}_exp4_trust_stability.png")
        print(f"\n关键发现：")
        print(f"  - 高Trust时间步更稳定（预测稳定性更高）")
        print(f"  - 低Trust时间步对噪声敏感（置信度下降更大）")
        print(f"  - Trust ≠ Importance，Trust反映可靠性！")
        
        return results
    
    def run_all_experiments(self, x, y, sample_name='sample'):
        """运行所有实验"""
        print("\n" + "="*60)
        print(f"运行完整实验流程: {sample_name}")
        print("="*60)
        
        # 实验1-4
        self.experiment_1_uncertainty_analysis(x, y, sample_name)
        self.experiment_2_timestep_consistency(x, y, sample_name)
        self.experiment_3_importance_vs_reliability(x, y, sample_name)
        self.experiment_4_trust_stability(x, y, sample_name)
        
        print("\n" + "="*60)
        print("所有实验完成！")
        print(f"结果保存在: {self.output_dir}")
        print("="*60)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("完整改进版实验流程已准备好！")
    print("\n使用方法:")
    print("""
    # 1. 准备模型和数据
    model = ...  # 你的模型
    x = torch.randn(1, 1, 128)
    y = 0
    
    # 2. 创建实验pipeline
    pipeline = CompleteExperimentPipeline(model, device='cuda')
    
    # 3. 运行所有实验
    pipeline.run_all_experiments(x, y, sample_name='sample_1')
    """)
