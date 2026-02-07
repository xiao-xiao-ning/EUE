"""
Visualization Tools - 可视化工具集
用于展示attribution、uncertainty、consistency和trust
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
import matplotlib.gridspec as gridspec


class AttributionVisualizer:
    """
    Attribution可视化工具
    """
    
    @staticmethod
    def plot_signal_with_attribution(
        signal: np.ndarray,
        attribution: np.ndarray,
        title: str = 'Time Series with Attribution',
        save_path: Optional[str] = None
    ):
        """
        绘制时间序列和对应的attribution
        
        Args:
            signal: [length] 或 [channels, length]
            attribution: [length]
            title: 图标题
            save_path: 保存路径
        """
        if signal.ndim == 2:
            signal = signal[0]  # 取第一个通道
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        
        # 信号
        axes[0].plot(signal, linewidth=1.5, color='steelblue')
        axes[0].set_ylabel('Signal Value', fontsize=11)
        axes[0].set_title(title, fontsize=13)
        axes[0].grid(True, alpha=0.3)
        
        # Attribution
        axes[1].bar(range(len(attribution)), attribution, 
                   color='coral', alpha=0.7, edgecolor='darkred')
        axes[1].set_xlabel('Time Step', fontsize=11)
        axes[1].set_ylabel('Attribution', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    
    @staticmethod
    def plot_attribution_heatmap(
        signal: np.ndarray,
        attribution: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
        title: str = 'Attribution Heatmap',
        save_path: Optional[str] = None
    ):
        """
        绘制attribution热力图
        
        Args:
            signal: [channels, length] 或 [length]
            attribution: [length]
            uncertainty: [length] (可选)
            title: 图标题
            save_path: 保存路径
        """
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        
        n_rows = 3 if uncertainty is not None else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, n_rows * 2))
        
        # 原始信号
        im1 = axes[0].imshow(signal, aspect='auto', cmap='viridis')
        axes[0].set_ylabel('Channel', fontsize=11)
        axes[0].set_title(f'{title} - Original Signal', fontsize=13)
        plt.colorbar(im1, ax=axes[0])
        
        # Attribution
        im2 = axes[1].imshow(attribution.reshape(1, -1), 
                            aspect='auto', cmap='RdBu_r', 
                            vmin=-np.abs(attribution).max(), 
                            vmax=np.abs(attribution).max())
        axes[1].set_ylabel('Attribution', fontsize=11)
        axes[1].set_xlabel('Time Step', fontsize=11)
        plt.colorbar(im2, ax=axes[1])
        
        # Uncertainty (如果提供)
        if uncertainty is not None:
            im3 = axes[2].imshow(uncertainty.reshape(1, -1), 
                                aspect='auto', cmap='YlOrRd')
            axes[2].set_ylabel('Uncertainty', fontsize=11)
            axes[2].set_xlabel('Time Step', fontsize=11)
            plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        

    @staticmethod
    def plot_multiview_attributions(
        views: Dict[str, np.ndarray],
        attributions: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """
        绘制多个view的attribution对比
        
        Args:
            views: {view_name: signal}
            attributions: {view_name: attribution}
            save_path: 保存路径
        """
        n_views = len(views)
        fig, axes = plt.subplots(n_views, 2, figsize=(16, 3 * n_views))
        
        if n_views == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (view_name, view_signal) in enumerate(views.items()):
            if view_signal.ndim == 2:
                view_signal = view_signal[0]
            
            attr = attributions[view_name]
            
            # 信号
            axes[idx, 0].plot(view_signal, linewidth=1.5)
            axes[idx, 0].set_ylabel('Signal', fontsize=10)
            axes[idx, 0].set_title(f'{view_name} - Signal', fontsize=11)
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Attribution
            axes[idx, 1].bar(range(len(attr)), attr, alpha=0.7)
            axes[idx, 1].set_ylabel('Attribution', fontsize=10)
            axes[idx, 1].set_title(f'{view_name} - Attribution', fontsize=11)
            axes[idx, 1].grid(True, alpha=0.3)
        
        axes[-1, 0].set_xlabel('Time Step', fontsize=11)
        axes[-1, 1].set_xlabel('Time Step', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        


class UncertaintyVisualizer:
    """
    Uncertainty可视化工具
    """
    
    @staticmethod
    def plot_mean_and_std(
        mean_attribution: np.ndarray,
        std_attribution: np.ndarray,
        signal: Optional[np.ndarray] = None,
        title: str = 'Attribution with Uncertainty',
        save_path: Optional[str] = None
    ):
        """
        绘制均值和标准差
        
        Args:
            mean_attribution: [length]
            std_attribution: [length]
            signal: [length] (可选)
            title: 图标题
            save_path: 保存路径
        """
        length = len(mean_attribution)
        x = np.arange(length)
        
        n_plots = 2 if signal is not None else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        # 信号 (如果提供)
        plot_idx = 0
        if signal is not None:
            if signal.ndim == 2:
                signal = signal[0]
            axes[plot_idx].plot(signal, linewidth=1.5, color='steelblue')
            axes[plot_idx].set_ylabel('Signal Value', fontsize=11)
            axes[plot_idx].set_title('Original Signal', fontsize=12)
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Attribution with uncertainty
        axes[plot_idx].plot(x, mean_attribution, linewidth=2, 
                           color='darkred', label='Mean Attribution')
        axes[plot_idx].fill_between(
            x,
            mean_attribution - std_attribution,
            mean_attribution + std_attribution,
            alpha=0.3,
            color='coral',
            label='±1 Std'
        )
        axes[plot_idx].fill_between(
            x,
            mean_attribution - 2 * std_attribution,
            mean_attribution + 2 * std_attribution,
            alpha=0.15,
            color='coral',
            label='±2 Std'
        )
        axes[plot_idx].axhline(y=0, color='gray', linestyle='--', linewidth=1)
        axes[plot_idx].set_xlabel('Time Step', fontsize=11)
        axes[plot_idx].set_ylabel('Attribution Value', fontsize=11)
        axes[plot_idx].set_title(title, fontsize=13)
        axes[plot_idx].legend(loc='best')
        axes[plot_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    
    @staticmethod
    def plot_uncertainty_regions(
        mean_attribution: np.ndarray,
        std_attribution: np.ndarray,
        threshold: float = 0.5,
        title: str = 'High/Low Uncertainty Regions',
        save_path: Optional[str] = None
    ):
        """
        标注高不确定性区域
        
        Args:
            mean_attribution: [length]
            std_attribution: [length]
            threshold: uncertainty阈值（归一化后）
            title: 图标题
            save_path: 保存路径
        """
        # 归一化uncertainty
        std_norm = std_attribution / (std_attribution.max() + 1e-8)
        high_uncertainty = std_norm > threshold
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        
        # Attribution
        axes[0].plot(mean_attribution, linewidth=2, color='darkblue')
        axes[0].fill_between(
            range(len(mean_attribution)),
            0,
            mean_attribution,
            where=high_uncertainty,
            alpha=0.4,
            color='red',
            label='High Uncertainty Regions'
        )
        axes[0].set_ylabel('Mean Attribution', fontsize=11)
        axes[0].set_title(title, fontsize=13)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Uncertainty
        axes[1].plot(std_norm, linewidth=2, color='darkred')
        axes[1].axhline(y=threshold, color='gray', linestyle='--', 
                       linewidth=1.5, label=f'Threshold={threshold}')
        axes[1].fill_between(
            range(len(std_norm)),
            0,
            std_norm,
            where=high_uncertainty,
            alpha=0.4,
            color='red'
        )
        axes[1].set_xlabel('Time Step', fontsize=11)
        axes[1].set_ylabel('Normalized Std', fontsize=11)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        


class TrustVisualizer:
    """
    Trust Score可视化工具
    """
    
    @staticmethod
    def plot_trust_comparison(
        signal: np.ndarray,
        mean_attribution: np.ndarray,
        trust_score: np.ndarray,
        trusted_attribution: np.ndarray,
        title: str = 'Trust-aware Attribution',
        save_path: Optional[str] = None
    ):
        """
        对比原始attribution和trust-aware attribution
        
        Args:
            signal: [length]
            mean_attribution: [length]
            trust_score: [length]
            trusted_attribution: [length]
            title: 图标题
            save_path: 保存路径
        """
        if signal.ndim == 2:
            signal = signal[0]
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])
        
        # 1. 原始信号
        ax0 = plt.subplot(gs[0])
        ax0.plot(signal, linewidth=1.5, color='steelblue')
        ax0.set_ylabel('Signal', fontsize=11)
        ax0.set_title(title, fontsize=14, fontweight='bold')
        ax0.grid(True, alpha=0.3)
        
        # 2. 原始Attribution
        ax1 = plt.subplot(gs[1])
        ax1.bar(range(len(mean_attribution)), mean_attribution, 
               color='coral', alpha=0.7, label='Original Attribution')
        ax1.set_ylabel('Attribution', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 3. Trust Score
        ax2 = plt.subplot(gs[2])
        ax2.plot(trust_score, linewidth=2, color='green', label='Trust Score')
        ax2.fill_between(range(len(trust_score)), 0, trust_score, 
                        alpha=0.3, color='green')
        ax2.set_ylabel('Trust', fontsize=11)
        ax2.set_ylim([0, 1.1])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 4. Trust-aware Attribution
        ax3 = plt.subplot(gs[3])
        ax3.bar(range(len(trusted_attribution)), trusted_attribution,
               color='darkgreen', alpha=0.7, label='Trust-weighted Attribution')
        ax3.set_xlabel('Time Step', fontsize=11)
        ax3.set_ylabel('Trusted Attr', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    
    @staticmethod
    def plot_trust_components(
        mean_attribution: np.ndarray,
        std_attribution: np.ndarray,
        consistency: float,
        trust_score: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        展示trust score的组成部分
        
        Args:
            mean_attribution: [length]
            std_attribution: [length]
            consistency: 标量
            trust_score: [length]
            save_path: 保存路径
        """
        # 归一化
        std_norm = std_attribution / (std_attribution.max() + 1e-8)
        uncertainty_component = 1 - std_norm
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        
        # 1. Uncertainty component
        axes[0].plot(uncertainty_component, linewidth=2, color='blue')
        axes[0].fill_between(range(len(uncertainty_component)), 0, 
                            uncertainty_component, alpha=0.3, color='blue')
        axes[0].set_ylabel('1 - Uncertainty', fontsize=11)
        axes[0].set_title('Trust Components Breakdown', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.1])
        
        # 2. Consistency component
        consistency_array = np.full_like(uncertainty_component, consistency)
        axes[1].plot(consistency_array, linewidth=2, color='orange')
        axes[1].fill_between(range(len(consistency_array)), 0, 
                            consistency_array, alpha=0.3, color='orange')
        axes[1].set_ylabel('Consistency', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.1])
        
        # 3. Final Trust Score
        axes[2].plot(trust_score, linewidth=2.5, color='darkgreen')
        axes[2].fill_between(range(len(trust_score)), 0, 
                            trust_score, alpha=0.3, color='darkgreen')
        axes[2].set_xlabel('Time Step', fontsize=11)
        axes[2].set_ylabel('Trust Score', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        


class ConsistencyVisualizer:
    """
    一致性可视化工具
    """
    
    @staticmethod
    def plot_consistency_matrix(
        attributions: Dict[str, np.ndarray],
        method: str = 'cosine',
        save_path: Optional[str] = None
    ):
        """
        绘制view之间的一致性矩阵
        
        Args:
            attributions: {view_name: attribution}
            method: 相似度计算方法
            save_path: 保存路径
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        view_names = list(attributions.keys())
        n_views = len(view_names)
        
        # 计算相似度矩阵
        similarity_matrix = np.zeros((n_views, n_views))
        
        for i in range(n_views):
            for j in range(n_views):
                attr_i = attributions[view_names[i]].flatten()
                attr_j = attributions[view_names[j]].flatten()
                
                if method == 'cosine':
                    sim = cosine_similarity(
                        attr_i.reshape(1, -1),
                        attr_j.reshape(1, -1)
                    )[0, 0]
                elif method == 'correlation':
                    sim = np.corrcoef(attr_i, attr_j)[0, 1]
                else:
                    sim = 0
                
                similarity_matrix[i, j] = sim
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=view_names,
            yticklabels=view_names,
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Similarity'}
        )
        plt.title(f'Cross-View Consistency Matrix ({method})', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        


# ===== 使用示例 =====

if __name__ == "__main__":
    print("Visualization Tools 已加载")
    print("\n可用的可视化类:")
    print("  - AttributionVisualizer: attribution基础可视化")
    print("  - UncertaintyVisualizer: uncertainty可视化")
    print("  - TrustVisualizer: trust score可视化")
    print("  - ConsistencyVisualizer: 跨视图一致性可视化")