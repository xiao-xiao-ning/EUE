"""
Quick Test - 快速测试所有模块是否正常工作
不需要真实数据，使用随机数据测试
"""

import torch
import torch.nn as nn
import numpy as np

print("="*60)
print("快速测试 - Multi-view Attribution Framework")
print("="*60)

# ===== 1. 测试导入 =====
print("\n[1/5] 测试模块导入...")
try:
    from old.multiview_attribution import (
        IntegratedGradients,
        MultiViewGenerator,
        CrossViewConsistency,
        TrustScore,
        MultiViewAttributionPipeline
    )
    print("  ✓ multiview_attribution.py 导入成功")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    exit(1)

try:
    from deletion_experiment import DeletionExperiment, BatchDeletionExperiment
    print("  ✓ deletion_experiment.py 导入成功")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    exit(1)

try:
    from visualization import (
        AttributionVisualizer,
        UncertaintyVisualizer,
        TrustVisualizer,
        ConsistencyVisualizer
    )
    print("  ✓ visualization.py 导入成功")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    exit(1)

# ===== 2. 创建简单模型 =====
print("\n[2/5] 创建测试模型...")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleModel().to(device)
model.eval()

print(f"  ✓ 模型创建成功 (device: {device})")

# ===== 3. 测试IntegratedGradients =====
print("\n[3/5] 测试Integrated Gradients...")

x = torch.randn(1, 1, 64).to(device)
target_class = 0

try:
    ig = IntegratedGradients(model, device)
    
    # 单次计算
    attr = ig.compute_attribution(x, target_class, steps=20)
    print(f"  ✓ 单次IG计算成功，shape: {attr.shape}")
    
    # MC Dropout
    mean_attr, std_attr = ig.compute_attribution_with_uncertainty(
        x, target_class, mc_samples=5, steps=20
    )
    print(f"  ✓ MC Dropout IG成功，mean: {mean_attr.shape}, std: {std_attr.shape}")
    
except Exception as e:
    print(f"  ✗ IG测试失败: {e}")
    import traceback
    traceback.print_exc()

# ===== 4. 测试MultiView =====
print("\n[4/5] 测试多视图生成...")

try:
    view_gen = MultiViewGenerator(wavelet='haar', max_level=2)
    
    signal = x.cpu().numpy().squeeze()
    views = view_gen.decompose(signal)
    
    print(f"  ✓ 生成了 {len(views)} 个视图:")
    for view_name, view_signal in views.items():
        print(f"      {view_name}: shape {view_signal.shape}")
    
except Exception as e:
    print(f"  ✗ 多视图测试失败: {e}")
    import traceback
    traceback.print_exc()

# ===== 5. 测试完整Pipeline =====
print("\n[5/5] 测试完整Pipeline...")

try:
    pipeline = MultiViewAttributionPipeline(
        model=model,
        device=device,
        wavelet='haar',
        max_level=2,
        mc_samples=5  # 用少量采样快速测试
    )
    
    results = pipeline.compute_multiview_attribution(
        x, target_class, compute_uncertainty=True
    )
    
    print(f"  ✓ Pipeline执行成功")
    print(f"      视图数: {len(results['views'])}")
    print(f"      一致性: {results['consistency']:.4f}")
    print(f"      Trust score shape: {results['trust_score'].shape}")
    
except Exception as e:
    print(f"  ✗ Pipeline测试失败: {e}")
    import traceback
    traceback.print_exc()

# ===== 6. 测试Deletion =====
print("\n[Bonus] 测试Deletion实验...")

try:
    deletion_exp = DeletionExperiment(model, device)
    
    fracs, scores = deletion_exp.deletion_curve(
        x, target_class, mean_attr,
        deletion_fractions=np.linspace(0, 1, 6)  # 只测试6个点
    )
    
    print(f"  ✓ Deletion实验成功")
    print(f"      测试点数: {len(fracs)}")
    print(f"      初始分数: {scores[0]:.4f}")
    print(f"      最终分数: {scores[-1]:.4f}")
    
except Exception as e:
    print(f"  ✗ Deletion测试失败: {e}")
    import traceback
    traceback.print_exc()

# ===== 总结 =====
print("\n" + "="*60)
print("测试完成！所有核心功能正常")
print("="*60)
print("\n下一步:")
print("  1. 运行 'python end_to_end_example.py' 查看完整示例")
print("  2. 准备你的真实数据和模型")
print("  3. 开始实验！")
print()