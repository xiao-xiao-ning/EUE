"""
Deletion Experiment - 验证Trust Score的有效性
通过逐步删除重要时间点来观察模型预测的变化
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm


class DeletionExperiment:
    """
    Deletion实验类
    用于验证attribution的质量
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def mask_timesteps(
        self,
        x: torch.Tensor,
        indices: np.ndarray,
        mask_value: float = 0.0
    ) -> torch.Tensor:
        """
        Mask指定的时间步
        
        Args:
            x: [1, channels, length]
            indices: 要mask的时间步索引
            mask_value: mask的值（0或均值）
            
        Returns:
            masked_x: [1, channels, length]
        """
        masked_x = x.clone()
        # 修复负stride问题：确保indices是连续的
        if isinstance(indices, np.ndarray):
            indices = indices.copy()  # 复制数组避免负stride
        masked_x[:, :, indices] = mask_value
        return masked_x
    
    def deletion_curve(
        self,
        x: torch.Tensor,
        target_class: int,
        attribution: np.ndarray,
        deletion_fractions: np.ndarray = np.linspace(0, 1, 21)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算deletion曲线
        
        Args:
            x: 输入样本 [1, channels, length]
            target_class: 目标类别
            attribution: 归因值 [length]
            deletion_fractions: 删除比例数组 [0, 0.05, 0.1, ..., 1.0]
            
        Returns:
            fractions: 删除比例
            scores: 对应的预测分数
        """
        x = x.to(self.device)
        length = x.shape[-1]
        
        # 按attribution绝对值排序（降序）
        sorted_indices = np.argsort(np.abs(attribution))[::-1]
        
        scores = []
        
        with torch.no_grad():
            for frac in deletion_fractions:
                # 计算要删除的时间步数量
                n_delete = int(frac * length)
                
                if n_delete == 0:
                    # 原始预测
                    output = self.model(x)
                    score = torch.softmax(output, dim=-1)[0, target_class].item()
                else:
                    # 删除top-n重要的时间步
                    indices_to_delete = sorted_indices[:n_delete]
                    masked_x = self.mask_timesteps(x, indices_to_delete)
                    
                    output = self.model(masked_x)
                    score = torch.softmax(output, dim=-1)[0, target_class].item()
                
                scores.append(score)
        
        return deletion_fractions, np.array(scores)
    
    def insertion_curve(
        self,
        x: torch.Tensor,
        target_class: int,
        attribution: np.ndarray,
        insertion_fractions: np.ndarray = np.linspace(0, 1, 21)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算insertion曲线（从全mask开始逐步插入重要时间步）
        
        Args:
            x: 输入样本 [1, channels, length]
            target_class: 目标类别
            attribution: 归因值 [length]
            insertion_fractions: 插入比例数组
            
        Returns:
            fractions: 插入比例
            scores: 对应的预测分数
        """
        x = x.to(self.device)
        length = x.shape[-1]
        
        # 按attribution绝对值排序（降序）
        sorted_indices = np.argsort(np.abs(attribution))[::-1]
        
        scores = []
        
        with torch.no_grad():
            for frac in insertion_fractions:
                # 从全mask开始
                masked_x = torch.zeros_like(x)
                
                # 计算要插入的时间步数量
                n_insert = int(frac * length)
                
                if n_insert > 0:
                    # 插入top-n重要的时间步
                    indices_to_insert = sorted_indices[:n_insert]
                    masked_x[:, :, indices_to_insert] = x[:, :, indices_to_insert]
                
                output = self.model(masked_x)
                score = torch.softmax(output, dim=-1)[0, target_class].item()
                
                scores.append(score)
        
        return insertion_fractions, np.array(scores)
    
    def compare_attributions(
        self,
        x: torch.Tensor,
        target_class: int,
        attributions_dict: Dict[str, np.ndarray],
        mode: str = 'deletion'
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        比较多个attribution方法
        
        Args:
            x: 输入样本
            target_class: 目标类别
            attributions_dict: {'method_name': attribution}
            mode: 'deletion' 或 'insertion'
            
        Returns:
            results: {'method_name': (fractions, scores)}
        """
        results = {}
        
        for method_name, attribution in attributions_dict.items():
            if mode == 'deletion':
                fractions, scores = self.deletion_curve(x, target_class, attribution)
            else:
                fractions, scores = self.insertion_curve(x, target_class, attribution)
            
            results[method_name] = (fractions, scores)
        
        return results
    
    @staticmethod
    def compute_auc(fractions: np.ndarray, scores: np.ndarray) -> float:
        """
        计算曲线下面积（使用梯形法则）
        
        Args:
            fractions: [n_points]
            scores: [n_points]
            
        Returns:
            auc: 标量
        """
        return np.trapezoid(scores, fractions)
    
    @staticmethod
    def plot_comparison(
        results: Dict[str, Tuple[np.ndarray, np.ndarray]],
        mode: str = 'deletion',
        save_path: str = None
    ):
        """
        可视化对比结果
        
        Args:
            results: {'method_name': (fractions, scores)}
            mode: 'deletion' 或 'insertion'
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))
        
        for method_name, (fractions, scores) in results.items():
            auc = DeletionExperiment.compute_auc(fractions, scores)
            plt.plot(
                fractions, 
                scores, 
                marker='o', 
                label=f'{method_name} (AUC={auc:.3f})'
            )
        
        plt.xlabel('Fraction of Timesteps Deleted/Inserted', fontsize=12)
        plt.ylabel('Target Class Probability', fontsize=12)
        plt.title(f'{mode.capitalize()} Curve Comparison', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')


class BatchDeletionExperiment:
    """
    批量Deletion实验
    用于在数据集上评估
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.deletion_exp = DeletionExperiment(model, device)
    
    def run_on_dataset(
        self,
        dataloader,
        attribution_methods: Dict[str, callable],
        mode: str = 'deletion',
        max_samples: int = None
    ) -> Dict[str, List[float]]:
        """
        在数据集上运行实验
        
        Args:
            dataloader: 数据加载器
            attribution_methods: {'method_name': compute_attribution_func}
            mode: 'deletion' 或 'insertion'
            max_samples: 最大样本数
            
        Returns:
            auc_scores: {'method_name': [auc1, auc2, ...]}
        """
        auc_scores = {name: [] for name in attribution_methods.keys()}
        
        sample_count = 0
        
        for batch_x, batch_y in tqdm(dataloader, desc='Running deletion experiments'):
            if max_samples and sample_count >= max_samples:
                break
            
            for i in range(len(batch_x)):
                x = batch_x[i:i+1]  # [1, channels, length]
                target_class = batch_y[i].item()
                
                # 对每个方法计算attribution
                attributions = {}
                for method_name, compute_fn in attribution_methods.items():
                    attr = compute_fn(x, target_class)
                    attributions[method_name] = attr
                
                # 运行deletion/insertion
                results = self.deletion_exp.compare_attributions(
                    x, target_class, attributions, mode
                )
                
                # 计算AUC
                for method_name, (fractions, scores) in results.items():
                    auc = self.deletion_exp.compute_auc(fractions, scores)
                    auc_scores[method_name].append(auc)
                
                sample_count += 1
                
                if max_samples and sample_count >= max_samples:
                    break
        
        return auc_scores
    
    @staticmethod
    def summarize_results(auc_scores: Dict[str, List[float]]) -> Dict[str, Dict]:
        """
        总结实验结果
        
        Args:
            auc_scores: {'method_name': [auc1, auc2, ...]}
            
        Returns:
            summary: {'method_name': {'mean': ..., 'std': ..., 'median': ...}}
        """
        summary = {}
        
        for method_name, scores in auc_scores.items():
            summary[method_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        return summary


def create_random_attribution_baseline(length: int) -> np.ndarray:
    """创建随机attribution作为baseline"""
    return np.random.randn(length)


def create_uniform_attribution_baseline(length: int) -> np.ndarray:
    """创建均匀attribution作为baseline"""
    return np.ones(length)


# ===== 使用示例 =====

if __name__ == "__main__":
    print("Deletion Experiment 模块已加载")
    print("\n使用示例:")
    print("""
    # 1. 单样本实验
    exp = DeletionExperiment(model, device='cuda')
    
    attributions = {
        'Original': original_attr,
        'Trust-weighted': trusted_attr,
        'Random': random_attr
    }
    
    results = exp.compare_attributions(x, target_class, attributions, mode='deletion')
    DeletionExperiment.plot_comparison(results, save_path='deletion_curve.png')
    
    # 2. 数据集批量实验
    batch_exp = BatchDeletionExperiment(model, device='cuda')
    
    attribution_methods = {
        'Original': lambda x, y: compute_original_attribution(x, y),
        'Trust': lambda x, y: compute_trust_attribution(x, y)
    }
    
    auc_scores = batch_exp.run_on_dataset(
        dataloader, 
        attribution_methods,
        max_samples=100
    )
    
    summary = BatchDeletionExperiment.summarize_results(auc_scores)
    print(summary)
    """)
