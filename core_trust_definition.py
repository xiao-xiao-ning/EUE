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
    
    这只是stability，不是trust！注意：这里的方差来源于解释过程中
    的推理随机性（如MC Dropout），并不等价于模型整体的epistemic
    uncertainty。
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
                当 metric='confidence_drop'（默认）时，ε 的语义对应于
                Softmax置信度的下降量。
            
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
            # 若解释未声称重要，则 Trust(t|x) 未定义，返回NaN
            return np.nan
        
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
    def _importance_gate(
        attributions: np.ndarray,
        threshold: float = 0.5,
        eps: float = 1e-8,
        normalize: bool = True
    ) -> np.ndarray:
        """Soft gating that activates only when |a_t| surpasses threshold."""
        abs_attr = np.abs(attributions)
        if normalize:
            # 将每个视图的attribution缩放到[0, 1]
            max_abs = abs_attr.max(axis=-1, keepdims=True)
            max_abs = np.where(max_abs < eps, eps, max_abs)
            abs_attr = abs_attr / max_abs
        gate = (abs_attr - threshold) / (1.0 - threshold + eps)
        return np.clip(gate, 0.0, 1.0)

    @staticmethod
    def compute_trust_aggregated(
        attributions_by_view,
        uncertainties_by_view,
        consistencies: np.ndarray,
        beta: float = 0.5,
        gamma: float = 0.5,
        importance_threshold: float = 0.5,
        gate_normalize: bool = True,
        alpha: float = 1.0
    ) -> np.ndarray:
        """
        Trust_agg(t) = E_r[ gate(|a_r(t)|) · exp(-beta U_r(t)) ] · Consistency(t)^gamma
        """
        if isinstance(attributions_by_view, dict):
            attr_stack = np.stack([np.abs(v) for v in attributions_by_view.values()])
        else:
            attr_stack = np.abs(np.asarray(attributions_by_view))
        if isinstance(uncertainties_by_view, dict):
            unc_stack = np.stack(list(uncertainties_by_view.values()))
        else:
            unc_stack = np.asarray(uncertainties_by_view)

        gate = TrustScore._importance_gate(
            attr_stack,
            threshold=importance_threshold,
            normalize=gate_normalize
        )
        if alpha != 1.0:
            gate = np.power(gate, alpha)
        reliability = np.exp(-beta * unc_stack)
        view_scores = gate * reliability
        mean_view_weight = np.nanmean(view_scores, axis=0)

        consistency_clamped = np.clip(consistencies, 0.0, 1.0)
        consistency_weight = np.power(consistency_clamped, gamma)
        trust = consistency_weight * mean_view_weight
        return np.clip(trust, 0.0, 1.0)
    
    @staticmethod
    def compute_trust_aggregated_normalized(
        attributions_by_view: dict,
        uncertainties_by_view: dict,
        consistencies: np.ndarray
    ) -> np.ndarray:
        return TrustScore.compute_trust_aggregated(
            attributions_by_view,
            uncertainties_by_view,
            consistencies
        )

    @staticmethod
    def compute_trusted_importance(
        attributions_by_view,
        trust: np.ndarray
    ) -> np.ndarray:
        if isinstance(attributions_by_view, dict):
            attr_stack = np.stack(
                [np.abs(attr) for attr in attributions_by_view.values()]
            )
        else:
            attr_stack = np.abs(np.asarray(attributions_by_view))
        mean_attr = attr_stack.mean(axis=0)
        return trust * mean_attr


# ==================== 时间步级别一致性 ====================

class TimestepConsistency:
    """
    Time-step-level Cross-view Consistency
    
    问题：不同视图在每个时间步上的attribution是否一致
    注意：这是时间步级别，不是全局相似度
    """
    
    @staticmethod
    # def compute_timestep_consistency(
    #     attributions: Dict[str, np.ndarray],  # {view_name: [length]}
    #     method: str = 'inverse_std'
    # ) -> np.ndarray:
    #     """
    #     计算每个时间步的跨视图一致性
        
    #     Args:
    #         attributions: {view_name: attribution[length]}
    #         method: 'inverse_std', 'inverse_range', 'inverse_cv'
            
    #     Returns:
    #         consistency: [length] 每个时间步的一致性
    #     """
    #     view_names = list(attributions.keys())
    #     length = attributions[view_names[0]].shape[0]
        
    #     consistency = np.zeros(length)
        
    #     for t in range(length):
    #         # 收集该时间步在所有view中的值
    #         values_at_t = np.array([attributions[v][t] for v in view_names])
            
    #         if method == 'inverse_std':
    #             # 标准差的倒数
    #             std = np.std(values_at_t)
    #             consistency[t] = 1.0 / (1.0 + std)
                
    #         elif method == 'inverse_range':
    #             # 范围的倒数
    #             value_range = np.ptp(values_at_t)  # peak-to-peak
    #             consistency[t] = 1.0 / (1.0 + value_range)
                
    #         elif method == 'inverse_cv':
    #             # 变异系数的倒数
    #             mean = np.mean(values_at_t)
    #             std = np.std(values_at_t)
    #             cv = std / (abs(mean) + 1e-8)
    #             consistency[t] = 1.0 / (1.0 + cv)
                
    #         elif method == 'cosine_global':
    #             stacked = np.stack([attributions[v] for v in view_names])
    #             norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    #             normalized = stacked / (norms + 1e-8)
    #             sim_matrix = normalized @ normalized.T
    #             mask = ~np.eye(len(view_names), dtype=bool)
    #             if mask.sum() == 0:
    #                 mean_sim = 1.0
    #             else:
    #                 mean_sim = sim_matrix[mask].mean()
    #             mean_sim = np.clip(mean_sim, 0.0, 1.0)
    #             consistency[:] = mean_sim
    #             break
    #         else:
    #             raise ValueError(f"Unknown method: {method}")
        
    #     return consistency

    def compute_timestep_consistency(attributions: Dict[str, np.ndarray], method: str = 'inverse_std') -> np.ndarray:
        view_names = list(attributions.keys())
        length = attributions[view_names[0]].shape[0]
        R = len(view_names)

        consistency = np.zeros(length)

        for t in range(length):
            vals = np.array([attributions[v][t] for v in view_names])

            # 1. sign agreement
            signs = np.sign(vals)
            agree = 0
            total = 0
            for i in range(R):
                for j in range(i + 1, R):
                    total += 1
                    agree += int(signs[i] == signs[j])
            sign_agreement = agree / (total + 1e-8)

            # 2. magnitude tolerance (log-scale)
            mags = np.abs(vals) + 1e-8
            mag_var = np.var(np.log(mags))
            mag_consistency = np.exp(-mag_var)

            consistency[t] = sign_agreement * mag_consistency
            consistency[t] = consistency[t] ** 0.5  # 平方根调整尺度

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