"""
Correct Trust Definition and Implementation
正确的Trust定义和实现

核心区别：
- Explanation Uncertainty: 归因本身的稳定性（attribution是否稳定）
- Trust: 归因声称的可验证性（当attribution说"重要"时，模型行为是否支持）

Trust ≠ Uncertainty！
Trust是在模型行为层面验证attribution的声称，而不是看attribution稳不稳。
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable
from scipy.stats import spearmanr


# ==================== 核心概念区分 ====================

class ExplanationUncertainty:
    """
    Explanation Uncertainty（解释不确定性）
    
    问题：这个重要性判断稳不稳？
    方法：多次计算attribution，看方差
    
    这只是stability，不是trust！
    """
    
    @staticmethod
    def compute_uncertainty(
        attributions: np.ndarray  # [n_samples, length]
    ) -> np.ndarray:
        """
        计算attribution的不确定性
        
        Args:
            attributions: [n_samples, length] 多次MC Dropout的结果
            
        Returns:
            uncertainty: [length] 每个时间步的标准差
        """
        return np.std(attributions, axis=0)


class TrustScore:
    """
    Trust Score（可信度评分）
    
    核心定义分为两部分：
    
    1. 基础Trust（通过扰动验证）：
       Trust(t|x) = E[𝟙(|f(x) - f(x\δ_t)| ≥ ε) | a_t ≥ τ]
       问题：当attribution说"t很重要"时，该不该信？
       方法：扰动时间点t，看模型输出是否真的变化
    
    2. 聚合Trust（整合多视图信息）：
       Trust_agg(t) = (1/R) Σ_r exp(-U_r(t)) · C_r(t) · A_r(t)
       
       其中：
       - R: 视图数量
       - U_r(t): 视图r在时间步t的不确定性（uncertainty）
       - C_r(t): 视图r在时间步t的一致性（consistency）
       - A_r(t): 视图r在时间步t的归因值（attribution）
       - exp(-U_r(t)): 不确定性的指数衰减，低不确定性 → 高权重
    
    Trust_agg综合考虑：
    - 低不确定性的视图权重更高（通过exp(-U)）
    - 高一致性的时间步更可信
    - 归因值本身的大小
    
    关键：
    1. 只在attribution声称"重要"时评估（条件化）
    2. 通过模型行为验证，不是看attribution稳定性
    3. Trust验证的是"重要性声称" vs "实际影响"
    4. Trust_agg整合了多视图、不确定性和一致性信息
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        epsilon: float = 0.1,  # 模型输出变化阈值
        importance_threshold: float = 0.5  # attribution重要性阈值
    ):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.importance_threshold = importance_threshold
        self.model.eval()
    
    def _perturb_timestep(
        self,
        x: torch.Tensor,
        t: int,
        perturbation_type: str = 'zero',
        magnitude: float = 0.2
    ) -> torch.Tensor:
        """
        扰动单个时间步（语义保持扰动）
        
        Args:
            x: [1, channels, length]
            t: 时间步索引
            perturbation_type: 'zero', 'noise', 'shuffle'
            magnitude: 扰动幅度
            
        Returns:
            perturbed_x: 扰动后的输入
        """
        x_pert = x.clone()
        
        if perturbation_type == 'zero':
            # 置零（最强扰动）
            x_pert[:, :, t] = 0
            
        elif perturbation_type == 'noise':
            # 加噪声
            noise = torch.randn_like(x[:, :, t]) * magnitude
            x_pert[:, :, t] += noise
            
        elif perturbation_type == 'shuffle':
            # 局部打乱（破坏时序）
            window = 5
            start = max(0, t - window // 2)
            end = min(x.shape[-1], t + window // 2 + 1)
            indices = torch.randperm(end - start) + start
            x_pert[:, :, start:end] = x[:, :, indices]
            
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
        
        return x_pert
    
    def _compute_output_change(
        self,
        x: torch.Tensor,
        x_perturbed: torch.Tensor,
        metric: str = 'prediction_change'
    ) -> float:
        """
        计算模型输出的变化程度
        
        Args:
            x: 原始输入
            x_perturbed: 扰动后的输入
            metric: 'prediction_change', 'confidence_drop', 'logit_diff'
            
        Returns:
            change: 输出变化量
        """
        with torch.no_grad():
            output_orig = self.model(x.to(self.device))
            output_pert = self.model(x_perturbed.to(self.device))
            
            if metric == 'prediction_change':
                # 预测是否改变（0或1）
                pred_orig = output_orig.argmax(dim=-1)
                pred_pert = output_pert.argmax(dim=-1)
                change = float(pred_orig != pred_pert)
                
            elif metric == 'confidence_drop':
                # 置信度下降
                prob_orig = torch.softmax(output_orig, dim=-1)
                prob_pert = torch.softmax(output_pert, dim=-1)
                target_class = output_orig.argmax(dim=-1)
                
                conf_orig = prob_orig[0, target_class].item()
                conf_pert = prob_pert[0, target_class].item()
                change = conf_orig - conf_pert
                
            elif metric == 'logit_diff':
                # Logit差异
                change = torch.abs(output_orig - output_pert).max().item()
                
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return change
    
    def compute_trust_single_timestep(
        self,
        x: torch.Tensor,
        t: int,
        attribution_value: float,
        n_perturbations: int = 20,
        perturbation_types: list = ['zero', 'noise']
    ) -> float:
        """
        计算单个时间步的trust
        
        Trust(t|x) = P(模型输出显著变化 | attribution说t重要)
        
        Args:
            x: 输入样本
            t: 时间步
            attribution_value: 该时间步的attribution值
            n_perturbations: 每种扰动类型的重复次数
            perturbation_types: 扰动类型列表
            
        Returns:
            trust: [0, 1] 可信度分数
        """
        # 归一化attribution值
        attr_norm = abs(attribution_value)
        
        # 只在attribution声称"重要"时计算trust
        if attr_norm < self.importance_threshold:
            # 如果attribution本身就说"不重要"，trust无意义
            return 0.0
        
        # 收集多次扰动的结果
        significant_changes = []
        
        for pert_type in perturbation_types:
            for _ in range(n_perturbations):
                # 扰动时间步t
                x_pert = self._perturb_timestep(x, t, pert_type)
                
                # 计算输出变化
                change = self._compute_output_change(
                    x, x_pert, metric='confidence_drop'
                )
                
                # 判断是否显著变化
                significant_changes.append(change >= self.epsilon)
        
        # Trust = 显著变化的比例
        trust = np.mean(significant_changes)
        
        return trust
    
    def compute_trust_all_timesteps(
        self,
        x: torch.Tensor,
        attribution: np.ndarray,
        n_perturbations: int = 5
    ) -> np.ndarray:
        """
        计算所有时间步的trust
        
        Args:
            x: [1, channels, length]
            attribution: [length]
            n_perturbations: 每个时间步的扰动次数
            
        Returns:
            trust_scores: [length]
        """
        length = attribution.shape[0]
        trust_scores = np.zeros(length)
        
        for t in range(length):
            trust_scores[t] = self.compute_trust_single_timestep(
                x, t, attribution[t], n_perturbations
            )
        
        return trust_scores
    
    @staticmethod
    def compute_trust_aggregated(
        attributions_by_view: dict,      # {view_name: attribution[length]}
        uncertainties_by_view: dict,     # {view_name: uncertainty[length]}
        consistencies: np.ndarray,       # [length] 时间步级别一致性
        use_exponential_decay: bool = True
    ) -> np.ndarray:
        """
        计算聚合Trust分数（Trust_agg）
        
        Trust_agg(t) = (1/R) Σ_r exp(-U_r(t)) · C_r(t) · A_r(t)
        
        Args:
            attributions_by_view: {view_name: attribution[length]}
            uncertainties_by_view: {view_name: uncertainty[length]}
            consistencies: [length] 每个时间步的跨视图一致性
            use_exponential_decay: 是否使用exp(-U)，否则用1/(1+U)
            
        Returns:
            trust_agg: [length] 聚合的trust分数
        """
        view_names = list(attributions_by_view.keys())
        R = len(view_names)  # 视图数量
        length = attributions_by_view[view_names[0]].shape[0]
        
        trust_agg = np.zeros(length)
        
        for t in range(length):
            weighted_sum = 0.0
            
            for view_name in view_names:
                A_r_t = attributions_by_view[view_name][t]  # 归因值
                U_r_t = uncertainties_by_view[view_name][t]  # 不确定性
                C_t = consistencies[t]  # 一致性（跨视图，所以所有视图共享）
                
                # 不确定性权重
                if use_exponential_decay:
                    # 指数衰减：低不确定性 → 高权重
                    uncertainty_weight = np.exp(-U_r_t)
                else:
                    # 倒数形式：也是低不确定性 → 高权重
                    uncertainty_weight = 1.0 / (1.0 + U_r_t)
                
                # 聚合：exp(-U_r(t)) · C_r(t) · A_r(t)
                # 注意：C_t是跨视图的，所以所有视图使用同一个C_t
                weighted_sum += uncertainty_weight * C_t * A_r_t
            
            # 平均
            trust_agg[t] = weighted_sum / R
        
        return trust_agg
    
    @staticmethod
    def compute_trust_aggregated_normalized(
        attributions_by_view: dict,
        uncertainties_by_view: dict,
        consistencies: np.ndarray
    ) -> np.ndarray:
        """
        计算归一化的Trust_agg（结果在[0,1]之间）
        
        Args:
            attributions_by_view: {view_name: attribution[length]}
            uncertainties_by_view: {view_name: uncertainty[length]}
            consistencies: [length]
            
        Returns:
            trust_agg_normalized: [length] 归一化的trust分数
        """
        # 先归一化各个组件
        view_names = list(attributions_by_view.keys())
        
        # 归一化attributions
        normalized_attr = {}
        for view_name in view_names:
            attr = attributions_by_view[view_name]
            attr_abs = np.abs(attr)
            attr_norm = attr_abs / (attr_abs.max() + 1e-8)
            normalized_attr[view_name] = attr_norm
        
        # 归一化uncertainties
        normalized_unc = {}
        for view_name in view_names:
            unc = uncertainties_by_view[view_name]
            unc_norm = unc / (unc.max() + 1e-8)
            normalized_unc[view_name] = unc_norm
        
        # consistency已经在[0,1]范围内
        
        # 计算trust_agg
        trust_agg = TrustScore.compute_trust_aggregated(
            normalized_attr,
            normalized_unc,
            consistencies,
            use_exponential_decay=True
        )
        
        # 归一化到[0,1]
        trust_agg_norm = trust_agg / (trust_agg.max() + 1e-8)
        
        return trust_agg_norm


# ==================== 时间步级别一致性 ====================

class TimestepConsistency:
    """
    Time-step-level Cross-view Consistency
    
    问题：不同视图在每个时间步上的attribution是否一致
    注意：这是时间步级别，不是全局相似度
    """
    
    @staticmethod
    def compute_timestep_consistency(
        attributions: Dict[str, np.ndarray],  # {view_name: [length]}
        method: str = 'inverse_std'
    ) -> np.ndarray:
        """
        计算每个时间步的跨视图一致性
        
        Args:
            attributions: {view_name: attribution[length]}
            method: 'inverse_std', 'inverse_range', 'inverse_cv'
            
        Returns:
            consistency: [length] 每个时间步的一致性
        """
        view_names = list(attributions.keys())
        length = attributions[view_names[0]].shape[0]
        
        consistency = np.zeros(length)
        
        for t in range(length):
            # 收集该时间步在所有view中的值
            values_at_t = np.array([attributions[v][t] for v in view_names])
            
            if method == 'inverse_std':
                # 标准差的倒数
                std = np.std(values_at_t)
                consistency[t] = 1.0 / (1.0 + std)
                
            elif method == 'inverse_range':
                # 范围的倒数
                value_range = np.ptp(values_at_t)  # peak-to-peak
                consistency[t] = 1.0 / (1.0 + value_range)
                
            elif method == 'inverse_cv':
                # 变异系数的倒数
                mean = np.mean(values_at_t)
                std = np.std(values_at_t)
                cv = std / (abs(mean) + 1e-8)
                consistency[t] = 1.0 / (1.0 + cv)
                
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return consistency


# ==================== 核心区分：三个概念 ====================

class ThreeConceptsDistinction:
    """
    明确区分三个概念：
    1. Attribution Importance（归因重要性）
    2. Explanation Uncertainty（解释不确定性）
    3. Trust（可信度）
    """
    
    @staticmethod
    def categorize_timesteps(
        attribution: np.ndarray,         # [length] 重要性
        uncertainty: np.ndarray,         # [length] 不确定性
        trust: np.ndarray,               # [length] 可信度
        importance_threshold: float = 0.5,
        trust_threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        将时间步分为八类（2×2×2）
        
        但最关键的对比是：
        - High importance + Low uncertainty + Low trust  → 稳定但不可信
        - High importance + High uncertainty + High trust → 不稳定但可信
        
        这证明了：Uncertainty ≠ Trust！
        """
        # 归一化
        attr_norm = np.abs(attribution)
        attr_norm = (attr_norm - attr_norm.min()) / (attr_norm.max() - attr_norm.min() + 1e-8)
        
        unc_norm = uncertainty / (uncertainty.max() + 1e-8)
        
        # 分类
        is_important = attr_norm > importance_threshold
        is_uncertain = unc_norm > 0.5  # 高不确定性
        is_trusted = trust > trust_threshold
        
        categories = {
            # 关键对比1：稳定但不可信
            'stable_but_untrusted': np.where(
                is_important & ~is_uncertain & ~is_trusted
            )[0],
            
            # 关键对比2：不稳定但可信
            'unstable_but_trusted': np.where(
                is_important & is_uncertain & is_trusted
            )[0],
            
            # 理想情况：稳定且可信
            'stable_and_trusted': np.where(
                is_important & ~is_uncertain & is_trusted
            )[0],
            
            # 最差情况：不稳定且不可信
            'unstable_and_untrusted': np.where(
                is_important & is_uncertain & ~is_trusted
            )[0],
            
            # 其他类别
            'unimportant_stable_trusted': np.where(
                ~is_important & ~is_uncertain & is_trusted
            )[0],
            
            'unimportant_stable_untrusted': np.where(
                ~is_important & ~is_uncertain & ~is_trusted
            )[0],
            
            'unimportant_unstable_trusted': np.where(
                ~is_important & is_uncertain & is_trusted
            )[0],
            
            'unimportant_unstable_untrusted': np.where(
                ~is_important & is_uncertain & ~is_trusted
            )[0],
        }
        
        return categories


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("="*70)
    print("正确的Trust定义和实现")
    print("="*70)
    
    print("\n核心区别:")
    print("-"*70)
    print("Explanation Uncertainty:")
    print("  问题: 这个重要性判断稳不稳？")
    print("  方法: 多次计算attribution，看方差")
    print("  本质: Attribution本身的stability")
    
    print("\nTrust (基础定义):")
    print("  问题: 当attribution说'重要'时，该不该信？")
    print("  方法: 扰动时间点，看模型输出是否真的变化")
    print("  本质: Attribution声称的可验证性（claim verification）")
    
    print("\nTrust_agg (聚合定义，推荐):")
    print("  公式: Trust_agg(t) = (1/R) Σ_r exp(-U_r(t)) · C_r(t) · A_r(t)")
    print("  整合: 多视图 + 不确定性 + 一致性 + 归因值")
    print("  优点: 计算快速，不需要额外扰动")
    
    print("\n关键洞察:")
    print("-"*70)
    print("❌ Low uncertainty ≠ High trust")
    print("✓ Trust_agg自动降低不确定视图的权重（exp(-U)）")
    print("✓ Trust_agg强调跨视图一致的时间步（C）")
    print("✓ Trust_agg考虑归因值大小（A）")
    
    print("\nTrust的两种计算方式:")
    print("-"*70)
    print("1. 扰动验证（精确但慢）:")
    print("   Trust(t|x) = E[𝟙(|f(x) - f(x\\δ_t)| ≥ ε) | a_t ≥ τ]")
    
    print("\n2. 聚合公式（快速且有效，推荐）:")
    print("   Trust_agg(t) = (1/R) Σ_r exp(-U_r(t)) · C_r(t) · A_r(t)")
    print("   - R: 视图数量")
    print("   - U_r(t): 视图r的不确定性")
    print("   - C_r(t): 时间步t的跨视图一致性")
    print("   - A_r(t): 视图r的归因值")
    
    print("\n" + "="*70)
    print("模块已加载，可以使用：")
    print("  - ExplanationUncertainty: 计算解释不确定性")
    print("  - TrustScore.compute_trust_all_timesteps: 扰动验证方法")
    print("  - TrustScore.compute_trust_aggregated: Trust_agg方法（推荐）")
    print("  - TimestepConsistency: 时间步级别一致性")
    print("  - ThreeConceptsDistinction: 三概念区分")
    print("="*70)