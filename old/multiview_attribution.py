"""
Multi-view Attribution with Uncertainty and Trust Score
用于时间序列分类的多视图可信解释框架
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import pywt
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


class MCDropoutWrapper(nn.Module):
    """
    MC Dropout包装器，用于启用测试时dropout
    """
    def __init__(self, model: nn.Module, dropout_rate: float = 0.2):
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self._enable_dropout()
    
    def _enable_dropout(self):
        """递归启用所有Dropout层"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # 强制dropout在eval模式下也工作
    
    def forward(self, x):
        return self.model(x)


class IntegratedGradients:
    """
    Integrated Gradients实现
    """
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
        """
        计算单个样本的Integrated Gradients
        
        Args:
            x: 输入样本 [1, channels, length]
            target_class: 目标类别
            baseline: 基线，默认为零基线
            steps: 积分步数
            
        Returns:
            attribution: [length] 归因值
        """
        x = x.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(self.device)
        
        # 生成插值路径
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        gradients = []
        for alpha in alphas:
            # 插值
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad_(True)
            
            # 前向传播
            output = self.model(interpolated)
            
            # 反向传播
            self.model.zero_grad()
            target_score = output[0, target_class]
            target_score.backward()
            
            # 保存梯度
            gradients.append(interpolated.grad.detach().cpu().numpy())
        
        # 平均梯度
        avg_gradients = np.mean(gradients, axis=0)
        
        # IG公式: (x - baseline) * avg_gradients
        integrated_gradients = (x - baseline).detach().cpu().numpy() * avg_gradients
        
        # 聚合到时间维度 [1, channels, length] -> [length]
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
        """
        使用MC Dropout计算attribution的均值和方差
        
        Args:
            x: 输入样本
            target_class: 目标类别
            mc_samples: MC采样次数
            
        Returns:
            mean_attribution: [length]
            std_attribution: [length]
        """
        attributions = []
        
        for _ in range(mc_samples):
            attr = self.compute_attribution(x, target_class, baseline, steps)
            attributions.append(attr)
        
        attributions = np.array(attributions)  # [mc_samples, length]
        
        mean_attr = np.mean(attributions, axis=0)
        std_attr = np.std(attributions, axis=0)
        
        return mean_attr, std_attr


class MultiViewGenerator:
    """
    多视图生成器 - 使用Haar小波变换
    """
    def __init__(self, wavelet: str = 'haar', max_level: int = 3):
        self.wavelet = wavelet
        self.max_level = max_level
    
    def decompose(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        小波分解生成多视图
        
        Args:
            signal: [length] 或 [channels, length]
            
        Returns:
            views: {'original', 'cA1', 'cD1', 'cA2', 'cD2', ...}
        """
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        
        views = {}
        views['original'] = signal
        
        # 对每个通道进行小波分解
        for level in range(1, self.max_level + 1):
            cA_list = []
            cD_list = []
            
            for channel in signal:
                coeffs = pywt.wavedec(channel, self.wavelet, level=level)
                cA = coeffs[0]  # 近似系数
                cD = coeffs[1] if len(coeffs) > 1 else np.zeros_like(cA)  # 细节系数
                
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
        """
        将不同view的attribution映射回时间域
        
        Args:
            attribution: view上的归因 [view_length]
            original_length: 原始信号长度
            view_name: 'original', 'cA1', 'cD1', etc.
            
        Returns:
            mapped_attribution: [original_length]
        """
        if view_name == 'original':
            return attribution
        
        # 简单的上采样映射
        # 更精确的方法可以使用小波重构
        mapped = np.interp(
            np.linspace(0, len(attribution), original_length),
            np.arange(len(attribution)),
            attribution
        )
        
        return mapped


class CrossViewConsistency:
    """
    跨视图一致性计算
    """
    @staticmethod
    def compute_consistency(
        attributions: Dict[str, np.ndarray],
        method: str = 'cosine'
    ) -> float:
        """
        计算多个view之间的一致性
        
        Args:
            attributions: {view_name: attribution [length]}
            method: 'cosine', 'correlation', 'rank_correlation'
            
        Returns:
            consistency_score: 标量
        """
        view_names = list(attributions.keys())
        n_views = len(view_names)
        
        if n_views < 2:
            return 1.0
        
        # 计算所有view pair的相似度
        similarities = []
        
        for i in range(n_views):
            for j in range(i + 1, n_views):
                attr_i = attributions[view_names[i]].flatten()
                attr_j = attributions[view_names[j]].flatten()
                
                if method == 'cosine':
                    sim = cosine_similarity(
                        attr_i.reshape(1, -1),
                        attr_j.reshape(1, -1)
                    )[0, 0]
                    
                elif method == 'correlation':
                    sim = np.corrcoef(attr_i, attr_j)[0, 1]
                    
                elif method == 'rank_correlation':
                    sim, _ = spearmanr(attr_i, attr_j)
                    
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                similarities.append(sim)
        
        # 平均相似度
        avg_consistency = np.mean(similarities)
        
        return avg_consistency
    
    @staticmethod
    def compute_timestep_consistency(
        attributions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        计算每个时间步的跨视图一致性
        
        Args:
            attributions: {view_name: attribution [length]}
            
        Returns:
            consistency: [length] 每个时间步的一致性
        """
        view_names = list(attributions.keys())
        length = attributions[view_names[0]].shape[0]
        
        consistency = np.zeros(length)
        
        for t in range(length):
            values = [attributions[v][t] for v in view_names]
            # 使用标准差的倒数作为一致性度量
            std = np.std(values)
            consistency[t] = 1.0 / (1.0 + std)
        
        return consistency


class TrustScore:
    """
    Trust Score计算器
    结合uncertainty和consistency
    """
    @staticmethod
    def compute_trust(
        mean_attribution: np.ndarray,
        std_attribution: np.ndarray,
        consistency: float,
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> np.ndarray:
        """
        计算trust score
        
        Trust = alpha * (1 - normalized_uncertainty) + beta * consistency
        
        Args:
            mean_attribution: [length]
            std_attribution: [length]
            consistency: 标量或 [length]
            alpha: uncertainty权重
            beta: consistency权重
            
        Returns:
            trust_score: [length]
        """
        # 归一化uncertainty (标准差)
        uncertainty_normalized = std_attribution / (np.max(std_attribution) + 1e-8)
        
        # 计算trust
        if isinstance(consistency, float):
            consistency = np.full_like(mean_attribution, consistency)
        
        trust = alpha * (1 - uncertainty_normalized) + beta * consistency
        
        return trust
    
    @staticmethod
    def get_trusted_attribution(
        mean_attribution: np.ndarray,
        trust_score: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        获取加权的可信归因
        
        Args:
            mean_attribution: [length]
            trust_score: [length]
            threshold: 可选的trust阈值
            
        Returns:
            trusted_attribution: [length]
        """
        if threshold is not None:
            mask = trust_score >= threshold
            trusted = mean_attribution * mask
        else:
            # 直接用trust加权
            trusted = mean_attribution * trust_score
        
        return trusted


class MultiViewAttributionPipeline:
    """
    完整的多视图归因流程
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        wavelet: str = 'haar',
        max_level: int = 2,
        mc_samples: int = 30
    ):
        self.device = device
        self.mc_samples = mc_samples
        
        # 用MC Dropout包装模型
        self.mc_model = MCDropoutWrapper(model)
        
        # 初始化组件
        self.ig = IntegratedGradients(self.mc_model, device)
        self.view_generator = MultiViewGenerator(wavelet, max_level)
        
    def compute_multiview_attribution(
        self,
        x: torch.Tensor,
        target_class: int,
        compute_uncertainty: bool = True
    ) -> Dict:
        """
        计算多视图归因
        
        Args:
            x: [1, channels, length]
            target_class: 目标类别
            compute_uncertainty: 是否计算不确定性
            
        Returns:
            results: {
                'views': {view_name: signal},
                'attributions': {view_name: {'mean': ..., 'std': ...}},
                'mapped_attributions': {view_name: attribution},
                'consistency': float,
                'trust_score': [length]
            }
        """
        x_np = x.cpu().numpy().squeeze()  # [channels, length]
        original_length = x_np.shape[-1]
        
        # 1. 生成多视图
        views = self.view_generator.decompose(x_np)
        
        # 2. 对每个view计算attribution
        attributions = {}
        mapped_attributions = {}
        
        for view_name, view_signal in views.items():
            # 转换为tensor
            view_tensor = torch.FloatTensor(view_signal).unsqueeze(0)  # [1, channels, length]
            
            if compute_uncertainty:
                mean_attr, std_attr = self.ig.compute_attribution_with_uncertainty(
                    view_tensor, target_class, self.mc_samples
                )
                attributions[view_name] = {
                    'mean': mean_attr,
                    'std': std_attr
                }
            else:
                attr = self.ig.compute_attribution(view_tensor, target_class)
                attributions[view_name] = {
                    'mean': attr,
                    'std': np.zeros_like(attr)
                }
            
            # 映射回时间域
            mapped = self.view_generator.map_to_time_domain(
                attributions[view_name]['mean'],
                original_length,
                view_name
            )
            mapped_attributions[view_name] = mapped
        
        # 3. 计算跨视图一致性
        consistency = CrossViewConsistency.compute_consistency(
            mapped_attributions,
            method='cosine'
        )
        
        # 4. 计算trust score (基于original view)
        mean_attr = attributions['original']['mean']
        std_attr = attributions['original']['std']
        
        trust_score = TrustScore.compute_trust(
            mean_attr,
            std_attr,
            consistency,
            alpha=0.5,
            beta=0.5
        )
        
        return {
            'views': views,
            'attributions': attributions,
            'mapped_attributions': mapped_attributions,
            'consistency': consistency,
            'trust_score': trust_score
        }


# ===== 辅助函数 =====

def normalize_attribution(attribution: np.ndarray) -> np.ndarray:
    """归一化attribution到[0, 1]"""
    attr_min = np.min(attribution)
    attr_max = np.max(attribution)
    if attr_max - attr_min < 1e-8:
        return np.zeros_like(attribution)
    return (attribution - attr_min) / (attr_max - attr_min)


def get_top_k_indices(attribution: np.ndarray, k: int) -> np.ndarray:
    """获取top-k重要的时间步索引"""
    return np.argsort(np.abs(attribution))[-k:]


if __name__ == "__main__":
    print("Multi-view Attribution Framework 已加载")
    print("主要组件:")
    print("  - IntegratedGradients: IG计算")
    print("  - MultiViewGenerator: 小波多视图生成")
    print("  - CrossViewConsistency: 一致性计算")
    print("  - TrustScore: 可信度评分")
    print("  - MultiViewAttributionPipeline: 完整流程")