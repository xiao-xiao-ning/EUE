"""
Unified Multi-view Trust-aware Attribution Framework
整合版多视图可信归因框架

整合了所有功能：
1. ✓ 正确的Trust定义（区别于Uncertainty）
2. ✓ Time-step-level Consistency
3. ✓ 支持多种模型（ResNet/Transformer）
4. ✓ 训练评估分离
5. ✓ 完整的实验流程
"""

import numpy as np
import torch
import torch.nn as nn
import pywt
from typing import Dict, Tuple, Optional, List
from pathlib import Path

# 导入核心组件
from core_trust_definition import (
    ExplanationUncertainty,
    TrustScore,
    TimestepConsistency,
    ThreeConceptsDistinction
)


# ==================== Integrated Gradients ====================

class MCDropoutWrapper(nn.Module):
    """MC Dropout包装器"""
    def __init__(self, model: nn.Module, dropout_rate: float = 0.2):
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self._enable_dropout()
    
    def _enable_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def forward(self, x):
        # Re-enable dropout before each forward so MC sampling stays stochastic
        self._enable_dropout()
        return self.model(x)


class IntegratedGradients:
    """Integrated Gradients计算"""
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def compute_attribution(
        self,
        x: torch.Tensor,
        target_class: int,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> np.ndarray:
        """计算单次IG"""
        x = x.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(self.device)
        
        alphas = torch.linspace(0, 1, steps).to(self.device)
        gradients = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad_(True)
            
            output = self.model(interpolated)
            self.model.zero_grad()
            target_score = output[0, target_class]
            target_score.backward()
            
            gradients.append(interpolated.grad.detach().cpu().numpy())
        
        avg_gradients = np.mean(gradients, axis=0)
        integrated_gradients = (x - baseline).detach().cpu().numpy() * avg_gradients
        attribution = np.sum(integrated_gradients, axis=1).squeeze()
        
        return attribution
    
    def compute_attribution_with_uncertainty(
        self,
        x: torch.Tensor,
        target_class: int,
        mc_samples: int = 30,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用MC Dropout计算attribution和uncertainty"""
        attributions = []
        
        for _ in range(mc_samples):
            attr = self.compute_attribution(x, target_class, baseline, steps)
            attributions.append(attr)
        
        attributions = np.array(attributions)
        
        mean_attr = np.mean(attributions, axis=0)
        std_attr = np.std(attributions, axis=0)
        
        return mean_attr, std_attr


# ==================== Multi-view Generator ====================

class MultiViewGenerator:
    """多视图生成器（Haar小波）"""
    def __init__(self, wavelet: str = 'haar', max_level: int = 2):
        self.wavelet = wavelet
        self.max_level = max_level
    
    def decompose(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """小波分解生成多视图"""
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        
        views = {}
        views['original'] = signal
        
        for level in range(1, self.max_level + 1):
            cA_list = []
            cD_list = []
            
            for channel in signal:
                coeffs = pywt.wavedec(channel, self.wavelet, level=level)
                cA = coeffs[0]
                cD = coeffs[1] if len(coeffs) > 1 else np.zeros_like(cA)
                
                cA_list.append(cA)
                cD_list.append(cD)
            
            views[f'cA{level}'] = np.array(cA_list)
            views[f'cD{level}'] = np.array(cD_list)
        
        return views
    
    def map_to_time_domain(
        self,
        attribution: np.ndarray,
        original_length: int,
        view_name: str
    ) -> np.ndarray:
        """映射回时间域"""
        if view_name == 'original':
            return attribution
        
        mapped = np.interp(
            np.linspace(0, len(attribution), original_length),
            np.arange(len(attribution)),
            attribution
        )
        
        return mapped


# ==================== 完整Pipeline ====================

class UnifiedMultiViewPipeline:
    """
    整合版完整Pipeline
    
    核心改进：
    1. 正确区分Uncertainty和Trust
    2. Time-step-level Consistency
    3. 完整的可信度评估
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        mc_samples: int = 30,
        trust_epsilon: float = 0.01,
        importance_threshold: float = 1e-4
    ):
        self.device = device
        self.mc_samples = mc_samples
        self.importance_threshold = importance_threshold
        
        # 用MC Dropout包装模型
        self.mc_model = MCDropoutWrapper(model)
        
        # 初始化组件
        self.ig = IntegratedGradients(self.mc_model, device)
        self.view_generator = MultiViewGenerator(wavelet='haar', max_level=2)
        
        # Trust计算（正确定义）
        self.trust_calculator = TrustScore(
            model=model,  # 注意：Trust用原始模型，不用MC Dropout
            device=device,
            epsilon=trust_epsilon,
            importance_threshold=importance_threshold
        )
        
        # Consistency计算
        self.consistency_calculator = TimestepConsistency()
        
        # Uncertainty计算
        self.uncertainty_calculator = ExplanationUncertainty()
    
    def compute_complete_explanation(
        self,
        x: torch.Tensor,
        target_class: int,
        compute_trust: bool = True,
        trust_n_perturbations: int = 10,
        trust_method: str = 'aggregated'  # 'aggregated' or 'perturbation'
    ) -> Dict:
        """
        计算完整的解释
        
        Args:
            x: 输入样本
            target_class: 目标类别
            compute_trust: 是否计算trust
            trust_n_perturbations: 扰动次数（仅perturbation方法使用）
            trust_method: Trust计算方法
                - 'aggregated': 使用Trust_agg公式（推荐）
                - 'perturbation': 使用扰动验证方法
        
        Returns:
            {
                'attribution_mean': [length],        # 重要性
                'attribution_std': [length],         # 不确定性（Uncertainty）
                'consistency': [length],             # 时间步级别一致性
                'trust': [length],                   # 可信度（Trust）
                'trust_method': str,                 # 使用的trust方法
                'categories': {...},                 # 时间步分类
                'views': {...},                      # 多视图
                'mapped_attributions': {...},        # 映射后的attribution
                'uncertainties_by_view': {...}       # 每个视图的uncertainty
            }
        """
        x_np = x.cpu().numpy().squeeze()
        original_length = x_np.shape[-1]
        
        # 1. 生成多视图
        views = self.view_generator.decompose(x_np)
        
        # 2. 对每个view计算attribution + uncertainty
        attributions_with_uncertainty = {}
        all_attributions = []
        
        for view_name, view_signal in views.items():
            view_tensor = torch.FloatTensor(view_signal).unsqueeze(0)
            
            mean_attr, std_attr = self.ig.compute_attribution_with_uncertainty(
                view_tensor, target_class, self.mc_samples
            )
            
            attributions_with_uncertainty[view_name] = {
                'mean': mean_attr,
                'std': std_attr
            }
            
            if view_name == 'original':
                all_attributions.append(mean_attr)
        
        # 3. 映射到时间域
        mapped_attributions = {}
        mapped_uncertainties = {}
        
        for view_name in views.keys():
            mapped_attr = self.view_generator.map_to_time_domain(
                attributions_with_uncertainty[view_name]['mean'],
                original_length,
                view_name
            )
            mapped_attributions[view_name] = mapped_attr
            
            mapped_unc = self.view_generator.map_to_time_domain(
                attributions_with_uncertainty[view_name]['std'],
                original_length,
                view_name
            )
            mapped_uncertainties[view_name] = mapped_unc
        
        # 4. 计算时间步级别一致性
        consistency = self.consistency_calculator.compute_timestep_consistency(
            mapped_attributions,
            method='inverse_std'
        )
        
        # 5. 提取original view的结果
        mean_attr = attributions_with_uncertainty['original']['mean']
        std_attr = attributions_with_uncertainty['original']['std']
        
        # 6. 计算Trust
        if compute_trust:
            if trust_method == 'aggregated':
                trust = TrustScore.compute_trust_aggregated(
                    mapped_attributions,
                    mapped_uncertainties,
                    consistency,
                    beta=0.5,
                    gamma=0.8,
                    importance_threshold=self.importance_threshold
                )
            elif trust_method == 'perturbation':
                # 使用扰动验证方法（计算较慢）
                trust = self.trust_calculator.compute_trust_all_timesteps(
                    x,
                    mean_attr,
                    n_perturbations=trust_n_perturbations
                )
            else:
                raise ValueError(f"Unknown trust_method: {trust_method}")
        else:
            trust = np.ones_like(mean_attr)
        
        # 7. 分类时间步（三概念区分）
        categories = ThreeConceptsDistinction.categorize_timesteps(
            mean_attr,
            std_attr,
            trust,
            importance_threshold=0.5,
            trust_threshold=0.5
        )
        
        trusted_importance = TrustScore.compute_trusted_importance(
            mapped_attributions,
            trust
        )

        return {
            'attribution_mean': mean_attr,
            'attribution_std': std_attr,
            'consistency': consistency,
            'trust': trust,
            'trusted_importance': trusted_importance,
            'trust_method': trust_method,
            'categories': categories,
            'views': views,
            'mapped_attributions': mapped_attributions,
            'uncertainties_by_view': mapped_uncertainties
        }


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    print("="*70)
    print("整合版Multi-view Trust-aware Attribution")
    print("="*70)
    
    # 创建示例模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 16, 3, padding=1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(16, 2)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.pool(x).squeeze(-1)
            return self.fc(x)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleModel().to(device)
    model.eval()
    
    # 创建pipeline
    pipeline = UnifiedMultiViewPipeline(
        model,
        device=device,
        mc_samples=10,  # 演示用少量采样
        trust_epsilon=0.1,
        importance_threshold=0.5
    )
    
    # 示例输入
    x = torch.randn(1, 1, 64).to(device)
    target_class = 0
    
    print("\n计算完整解释...")
    results = pipeline.compute_complete_explanation(
        x, target_class,
        compute_trust=True,
        trust_n_perturbations=3  # 演示用少量扰动
    )
    
    print("\n结果摘要:")
    print(f"  Attribution shape: {results['attribution_mean'].shape}")
    print(f"  Uncertainty shape: {results['attribution_std'].shape}")
    print(f"  Consistency shape: {results['consistency'].shape}")
    print(f"  Trust shape: {results['trust'].shape}")
    
    print("\n时间步分类:")
    for cat_name, indices in results['categories'].items():
        if len(indices) > 0:
            print(f"  {cat_name}: {len(indices)} 个时间步")
    
    print("\n关键区别:")
    print("  Uncertainty (attribution稳定性): "
          f"mean={results['attribution_std'].mean():.4f}")
    print("  Trust (声称可验证性): "
          f"mean={results['trust'].mean():.4f}")
    
    print("\n" + "="*70)
    print("✓ Pipeline执行成功！")
    print("="*70)


if __name__ == "__main__":
    example_usage()