"""
Enhanced Consistency and Trust Evaluation
增强版一致性计算和Trust评估

核心改进：
1. Time-step-level consistency（不是global view-level）
2. 明确区分 attribution importance vs explanation reliability (trust)
3. 设计实验验证：高trust ≠ 高attribution，而是更稳定、更可靠
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity


class TimestepLevelConsistency:
    """
    时间步级别的跨视图一致性计算
    
    关键：不是计算全局的view-level相似度，
    而是对每个时间步单独计算其在不同视图下的一致性
    """
    
    @staticmethod
    def compute_timestep_consistency(
        attributions: Dict[str, np.ndarray],
        method: str = 'std'
    ) -> np.ndarray:
        """
        计算每个时间步的跨视图一致性
        
        Args:
            attributions: {view_name: attribution [length]}
            method: 'std' (标准差倒数), 'range' (范围倒数), 'cv' (变异系数倒数)
            
        Returns:
            consistency: [length] 每个时间步的一致性分数
        
        示例：
            view1: [0.5, 0.8, 0.3]  # 3个时间步
            view2: [0.6, 0.7, 0.1]
            view3: [0.5, 0.9, 0.2]
            
            时间步0: [0.5, 0.6, 0.5] → std=0.058 → consistency高
            时间步1: [0.8, 0.7, 0.9] → std=0.100 → consistency中等
            时间步2: [0.3, 0.1, 0.2] → std=0.100 → consistency中等
        """
        view_names = list(attributions.keys())
        length = attributions[view_names[0]].shape[0]
        
        consistency = np.zeros(length)
        
        for t in range(length):
            # 收集该时间步在所有view中的值
            values_at_t = np.array([attributions[v][t] for v in view_names])
            
            if method == 'std':
                # 标准差倒数（值越接近，std越小，consistency越高）
                std = np.std(values_at_t)
                consistency[t] = 1.0 / (1.0 + std)
                
            elif method == 'range':
                # 范围倒数
                value_range = np.max(values_at_t) - np.min(values_at_t)
                consistency[t] = 1.0 / (1.0 + value_range)
                
            elif method == 'cv':
                # 变异系数倒数 (coefficient of variation)
                mean = np.mean(values_at_t)
                std = np.std(values_at_t)
                cv = std / (abs(mean) + 1e-8)
                consistency[t] = 1.0 / (1.0 + cv)
                
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return consistency
    
    @staticmethod
    def compute_pairwise_timestep_agreement(
        attr1: np.ndarray,
        attr2: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        计算两个attribution在时间步级别的agreement
        
        Args:
            attr1, attr2: [length] 两个视图的attribution
            threshold: 归一化后的阈值，用于判断"重要"时间步
            
        Returns:
            agreement: [0, 1] 两个视图在重要时间步上的一致性
        """
        # 归一化
        attr1_norm = (attr1 - attr1.min()) / (attr1.max() - attr1.min() + 1e-8)
        attr2_norm = (attr2 - attr2.min()) / (attr2.max() - attr2.min() + 1e-8)
        
        # 识别重要时间步
        important_1 = attr1_norm > threshold
        important_2 = attr2_norm > threshold
        
        # 计算overlap
        intersection = np.logical_and(important_1, important_2).sum()
        union = np.logical_or(important_1, important_2).sum()
        
        agreement = intersection / (union + 1e-8)
        
        return agreement


class ImportanceVsReliability:
    """
    明确区分 Attribution Importance 和 Explanation Reliability (Trust)
    
    关键概念：
    - Importance: 该时间步对预测的影响程度（高attribution值）
    - Reliability: 该解释的可信程度（低uncertainty + 高consistency）
    
    四种情况：
    1. 高importance + 高reliability → 真正的关键时间步 ✓
    2. 高importance + 低reliability → 不可靠的虚假重要性 ✗
    3. 低importance + 高reliability → 确实不重要 ✓
    4. 低importance + 低reliability → 不确定 ?
    """
    
    @staticmethod
    def categorize_timesteps(
        attribution: np.ndarray,
        trust: np.ndarray,
        importance_threshold: float = 0.5,
        trust_threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        将时间步分为四类
        
        Args:
            attribution: [length] attribution值
            trust: [length] trust分数
            importance_threshold: importance阈值（归一化后）
            trust_threshold: trust阈值
            
        Returns:
            categories: {
                'reliable_important': indices,     # 可信的重要点
                'unreliable_important': indices,   # 不可信的重要点
                'reliable_unimportant': indices,   # 可信的不重要点
                'unreliable_unimportant': indices  # 不确定点
            }
        """
        # 归一化importance
        attr_norm = np.abs(attribution)
        attr_norm = (attr_norm - attr_norm.min()) / (attr_norm.max() - attr_norm.min() + 1e-8)
        
        # 分类
        is_important = attr_norm > importance_threshold
        is_reliable = trust > trust_threshold
        
        categories = {
            'reliable_important': np.where(is_important & is_reliable)[0],
            'unreliable_important': np.where(is_important & ~is_reliable)[0],
            'reliable_unimportant': np.where(~is_important & is_reliable)[0],
            'unreliable_unimportant': np.where(~is_important & ~is_reliable)[0]
        }
        
        return categories
    
    @staticmethod
    def compute_reliability_metrics(
        uncertainty: np.ndarray,
        consistency: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        计算纯粹的reliability分数（不考虑importance）
        
        Args:
            uncertainty: [length] 每个时间步的uncertainty
            consistency: [length] 每个时间步的consistency
            alpha: uncertainty权重
            
        Returns:
            reliability: [length] 可靠性分数
        """
        # 归一化uncertainty（越低越好）
        uncertainty_norm = uncertainty / (uncertainty.max() + 1e-8)
        
        # reliability = 低uncertainty + 高consistency
        reliability = alpha * (1 - uncertainty_norm) + (1 - alpha) * consistency
        
        return reliability


class TrustStabilityExperiment:
    """
    实验设计：验证 Trust 不等于 Importance，而是反映稳定性和可靠性
    
    核心思想：
    1. 对输入加噪声
    2. 观察高importance但低trust的时间步 vs 高trust的时间步
    3. 预期：高trust的时间步在噪声下更稳定
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def add_noise_to_timesteps(
        self,
        x: torch.Tensor,
        timestep_indices: np.ndarray,
        noise_level: float = 0.1
    ) -> torch.Tensor:
        """
        对指定时间步添加噪声
        
        Args:
            x: [1, channels, length]
            timestep_indices: 要加噪声的时间步
            noise_level: 噪声强度
            
        Returns:
            noisy_x: 加噪后的输入
        """
        noisy_x = x.clone()
        
        # 计算噪声
        noise = torch.randn_like(x[:, :, timestep_indices]) * noise_level
        noisy_x[:, :, timestep_indices] += noise
        
        return noisy_x
    
    def stability_under_noise(
        self,
        x: torch.Tensor,
        target_class: int,
        timestep_indices: np.ndarray,
        noise_levels: List[float] = [0.05, 0.1, 0.2, 0.3],
        n_trials: int = 10
    ) -> Dict[str, float]:
        """
        测试指定时间步在噪声下的稳定性
        
        Args:
            x: 输入样本
            target_class: 目标类别
            timestep_indices: 要测试的时间步
            noise_levels: 噪声水平列表
            n_trials: 每个噪声水平的重复次数
            
        Returns:
            stability_metrics: {
                'prediction_stability': 预测稳定性,
                'confidence_drop': 置信度下降,
                'mean_probability_std': 概率标准差
            }
        """
        original_output = self.model(x.to(self.device))
        original_prob = torch.softmax(original_output, dim=-1)[0, target_class].item()
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for noise_level in noise_levels:
                for _ in range(n_trials):
                    # 加噪声
                    noisy_x = self.add_noise_to_timesteps(x, timestep_indices, noise_level)
                    
                    # 预测
                    output = self.model(noisy_x.to(self.device))
                    prob = torch.softmax(output, dim=-1)
                    
                    pred = output.argmax().item()
                    prob_target = prob[0, target_class].item()
                    
                    predictions.append(pred == target_class)
                    probabilities.append(prob_target)
        
        # 计算稳定性指标
        prediction_stability = np.mean(predictions)  # 预测不变的比例
        confidence_drop = original_prob - np.mean(probabilities)  # 置信度下降
        probability_std = np.std(probabilities)  # 概率波动
        
        return {
            'prediction_stability': prediction_stability,
            'confidence_drop': confidence_drop,
            'probability_std': probability_std
        }
    
    def compare_high_importance_vs_high_trust(
        self,
        x: torch.Tensor,
        target_class: int,
        categories: Dict[str, np.ndarray],
        noise_levels: List[float] = [0.1, 0.2, 0.3]
    ) -> Dict:
        """
        对比实验：高importance但低trust vs 高trust的时间步
        
        预期结果：
        - reliable_important: 高稳定性（真正的关键点）
        - unreliable_important: 低稳定性（虚假的重要性）
        
        这证明了：trust ≠ importance，trust反映可靠性
        """
        results = {}
        
        for category_name, indices in categories.items():
            if len(indices) == 0:
                continue
            
            stability = self.stability_under_noise(
                x, target_class, indices, noise_levels
            )
            
            results[category_name] = stability
        
        return results


# ==================== 使用示例 ====================

def example_timestep_consistency():
    """示例：时间步级别一致性"""
    print("="*60)
    print("示例1：时间步级别一致性计算")
    print("="*60)
    
    # 模拟3个view的attribution
    attributions = {
        'original': np.array([0.5, 0.8, 0.3, 0.1, 0.6]),
        'cA1': np.array([0.6, 0.7, 0.2, 0.1, 0.5]),
        'cD1': np.array([0.5, 0.9, 0.3, 0.2, 0.7])
    }
    
    # 计算时间步级别一致性
    consistency = TimestepLevelConsistency.compute_timestep_consistency(
        attributions, method='std'
    )
    
    print("\n各视图的attribution:")
    for view, attr in attributions.items():
        print(f"  {view}: {attr}")
    
    print(f"\n时间步级别一致性: {consistency}")
    print(f"  时间步0 (值接近): {consistency[0]:.3f} ← 高一致性")
    print(f"  时间步1 (值分散): {consistency[1]:.3f} ← 低一致性")


def example_importance_vs_reliability():
    """示例：区分importance和reliability"""
    print("\n" + "="*60)
    print("示例2：区分Attribution Importance和Explanation Reliability")
    print("="*60)
    
    # 模拟数据
    length = 10
    attribution = np.random.randn(length)
    uncertainty = np.random.rand(length) * 0.3  # [0, 0.3]
    consistency = np.random.rand(length) * 0.5 + 0.5  # [0.5, 1.0]
    
    # 计算reliability（不考虑importance）
    reliability = ImportanceVsReliability.compute_reliability_metrics(
        uncertainty, consistency, alpha=0.5
    )
    
    # 分类时间步
    categories = ImportanceVsReliability.categorize_timesteps(
        attribution, reliability,
        importance_threshold=0.5,
        trust_threshold=0.6
    )
    
    print("\n时间步分类:")
    for category, indices in categories.items():
        print(f"  {category}: {len(indices)} 个时间步")
        if len(indices) > 0:
            print(f"    索引: {indices[:5]}...")  # 只显示前5个
    
    print("\n关键区别:")
    print("  - reliable_important: 真正的关键时间步 ✓")
    print("  - unreliable_important: 虚假的重要性（需要警惕）✗")
    print("  - reliability ≠ importance，reliability表示可信度")


def example_trust_stability():
    """示例：Trust稳定性实验"""
    print("\n" + "="*60)
    print("示例3：Trust稳定性实验设计")
    print("="*60)
    
    print("""
    实验设计：
    
    1. 选择两组时间步：
       - Group A: 高importance + 高trust (可信的重要点)
       - Group B: 高importance + 低trust (不可信的重要点)
    
    2. 对这两组分别加噪声，观察模型预测变化
    
    3. 预期结果：
       - Group A: 预测稳定（置信度变化小）
       - Group B: 预测不稳定（置信度大幅下降）
    
    4. 结论：
       - Trust高的时间步确实更可靠（噪声鲁棒）
       - Trust ≠ Importance，而是反映解释的可信度
    
    这个实验可以放在论文的 4.4 Trust-aware Explanation Evaluation 部分
    """)


if __name__ == "__main__":
    example_timestep_consistency()
    example_importance_vs_reliability()
    example_trust_stability()
    
    print("\n" + "="*60)
    print("增强版一致性和Trust评估模块就绪！")
    print("="*60)
    print("\n关键改进:")
    print("  1. ✓ Time-step-level consistency（不是global）")
    print("  2. ✓ 明确区分 importance vs reliability")
    print("  3. ✓ Trust稳定性实验设计")
