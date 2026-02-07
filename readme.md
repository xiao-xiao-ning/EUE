# Multi-view Trust-aware Attribution Framework

用于时间序列分类的多视图可信解释框架。该框架通过以下三个核心组件实现可信的模型解释：

1. **Explanation Uncertainty** - 量化单个解释的不确定性
2. **Cross-view Consistency** - 评估多视图解释的一致性
3. **Trust Score** - 综合uncertainty和consistency的可信度评分

## 项目结构

```
.
├── multiview_attribution.py    # 核心框架：IG计算、多视图生成、trust计算
├── deletion_experiment.py      # Deletion/Insertion实验
├── visualization.py            # 可视化工具集
├── end_to_end_example.py       # 完整示例
├── requirements.txt            # 依赖包
└── README.md                   # 本文档
```

## 核心组件说明

### 1. `multiview_attribution.py`

**主要类：**

- **`IntegratedGradients`** - IG归因计算
  - `compute_attribution()` - 单次IG计算
  - `compute_attribution_with_uncertainty()` - MC Dropout多次采样

- **`MultiViewGenerator`** - 小波多视图生成
  - `decompose()` - Haar小波分解（cA1, cD1, cA2, cD2...）
  - `map_to_time_domain()` - 映射回时间域

- **`CrossViewConsistency`** - 跨视图一致性
  - `compute_consistency()` - 全局一致性（cosine/correlation）
  - `compute_timestep_consistency()` - 时间步级一致性

- **`TrustScore`** - 可信度计算
  - `compute_trust()` - 组合uncertainty和consistency
  - `get_trusted_attribution()` - Trust加权的attribution

- **`MultiViewAttributionPipeline`** - 完整流程封装

**使用示例：**

```python
from multiview_attribution import MultiViewAttributionPipeline

pipeline = MultiViewAttributionPipeline(
    model=your_model,
    device='cuda',
    wavelet='haar',
    max_level=2,
    mc_samples=30
)

results = pipeline.compute_multiview_attribution(
    x=input_tensor,  # [1, channels, length]
    target_class=predicted_class,
    compute_uncertainty=True
)

# 提取结果
trust_score = results['trust_score']
consistency = results['consistency']
attributions = results['attributions']
```

### 2. `deletion_experiment.py`

**主要类：**

- **`DeletionExperiment`** - 单样本deletion/insertion
  - `deletion_curve()` - 逐步删除重要时间步
  - `insertion_curve()` - 逐步插入重要时间步
  - `compare_attributions()` - 对比多个attribution方法

- **`BatchDeletionExperiment`** - 批量实验
  - `run_on_dataset()` - 在数据集上运行
  - `summarize_results()` - 统计结果

**使用示例：**

```python
from deletion_experiment import DeletionExperiment

exp = DeletionExperiment(model, device='cuda')

attributions_dict = {
    'Original': original_attr,
    'Trust-weighted': trusted_attr,
    'Random': random_attr
}

results = exp.compare_attributions(
    x, target_class, attributions_dict, mode='deletion'
)

DeletionExperiment.plot_comparison(results, save_path='deletion.png')
```

### 3. `visualization.py`

**主要类：**

- **`AttributionVisualizer`** - Attribution基础可视化
- **`UncertaintyVisualizer`** - Uncertainty可视化
- **`TrustVisualizer`** - Trust score可视化
- **`ConsistencyVisualizer`** - 一致性矩阵

**使用示例：**

```python
from visualization import TrustVisualizer

TrustVisualizer.plot_trust_comparison(
    signal=time_series,
    mean_attribution=mean_attr,
    trust_score=trust,
    trusted_attribution=trusted_attr,
    save_path='trust_comparison.png'
)
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行完整示例

```bash
python end_to_end_example.py
```

这会：
- 创建合成数据集
- 训练一个简单的ResNet模型
- 对测试样本计算multi-view attribution
- 运行deletion实验
- 生成所有可视化

### 3. 在你自己的模型上使用

```python
import torch
from multiview_attribution import MultiViewAttributionPipeline

# 1. 准备你的模型和数据
your_model = ...  # 已训练的PyTorch模型
your_model.eval()

x = torch.randn(1, 1, 128)  # [batch=1, channels, length]
target_class = 0

# 2. 创建pipeline
pipeline = MultiViewAttributionPipeline(
    model=your_model,
    device='cuda',
    mc_samples=30  # MC Dropout采样次数
)

# 3. 计算trust-aware attribution
results = pipeline.compute_multiview_attribution(
    x, target_class, compute_uncertainty=True
)

# 4. 使用结果
print(f"跨视图一致性: {results['consistency']:.4f}")
print(f"Trust score: {results['trust_score']}")
```

## 实验设计建议

根据你的论文实验设计，建议按以下顺序进行：

### 实验1：Explanation Uncertainty分析

**目标：** 验证attribution uncertainty不是噪声

```python
# 对同一样本多次计算attribution
mean_attr, std_attr = pipeline.ig.compute_attribution_with_uncertainty(
    x, target_class, mc_samples=30
)

# 可视化
from visualization import UncertaintyVisualizer
UncertaintyVisualizer.plot_mean_and_std(mean_attr, std_attr, signal=x)
```

**展示：**
- 高attribution + 低variance → 关键区域
- 高variance → 不可靠区域

### 实验2：Cross-view Consistency评估

**目标：** 验证不同视图在关键时间段更一致

```python
results = pipeline.compute_multiview_attribution(x, target_class)
consistency = results['consistency']
mapped_attributions = results['mapped_attributions']

# 可视化一致性矩阵
from visualization import ConsistencyVisualizer
ConsistencyVisualizer.plot_consistency_matrix(mapped_attributions)
```

### 实验3：Trust-aware Deletion实验（最重要）

**目标：** 验证trust能识别"重要但不可靠"的解释

```python
from deletion_experiment import DeletionExperiment

exp = DeletionExperiment(model, device='cuda')

attributions = {
    'Original': mean_attr,
    'Trust-weighted': trusted_attr,
    'Random': random_baseline
}

results = exp.compare_attributions(x, target_class, attributions)
DeletionExperiment.plot_comparison(results)
```

**预期结果：**
- Trust-weighted deletion曲线下降更快
- AUC更高 → 选中的时间点更"可信地重要"

### 实验4：Ablation Study

```python
# 测试不同组合
trust_only_uncertainty = TrustScore.compute_trust(
    mean_attr, std_attr, 
    consistency=1.0,  # 忽略consistency
    alpha=1.0, beta=0.0
)

trust_only_consistency = TrustScore.compute_trust(
    mean_attr, 
    std_attr=np.zeros_like(std_attr),  # 忽略uncertainty
    consistency=consistency,
    alpha=0.0, beta=1.0
)

trust_full = TrustScore.compute_trust(
    mean_attr, std_attr, consistency,
    alpha=0.5, beta=0.5
)
```

## 在UCR数据集上使用

```python
from tslearn.datasets import UCR_UEA_datasets

# 加载UCR数据集
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ECG200')

# 转换为PyTorch格式
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 运行实验...
```

## 输出说明

运行示例后，`./experiment_outputs/` 目录包含：

- `test_sample_0_basic.png` - 基础attribution可视化
- `test_sample_0_uncertainty.png` - Uncertainty可视化
- `test_sample_0_trust.png` - Trust对比可视化
- `test_sample_0_multiview.png` - 多视图对比
- `test_sample_0_consistency.png` - 一致性矩阵
- `test_sample_0_deletion.png` - Deletion曲线
- `batch_evaluation.png` - 批量评估结果

## 关键参数说明

### MultiViewAttributionPipeline

- `wavelet`: 小波类型，默认'haar'
- `max_level`: 小波分解层数，推荐2-3
- `mc_samples`: MC Dropout采样次数，推荐20-50

### TrustScore

- `alpha`: Uncertainty权重（0-1）
- `beta`: Consistency权重（0-1）
- 建议 alpha=beta=0.5

### DeletionExperiment

- `deletion_fractions`: 删除比例，默认np.linspace(0, 1, 21)
- `mask_value`: mask值，默认0.0

## 论文写作建议

1. **Method Section**
   - 描述MC Dropout如何量化uncertainty
   - 解释Haar小波如何生成多视图
   - 给出Trust score的数学公式

2. **Experiments Section**
   - 4.2 Explanation Uncertainty Analysis
   - 4.3 Cross-view Consistency Evaluation
   - 4.4 Trust-aware Explanation Evaluation (Deletion)
   - 4.5 Ablation Study

3. **结果展示**
   - 定性：可视化图（uncertainty, trust对比）
   - 定量：Deletion AUC表格

## 常见问题

**Q: MC Dropout采样次数如何选择？**
A: 20-50次足够，更多会增加计算时间但收益递减

**Q: 小波层数如何选择？**
A: 2-3层足够，太深会导致分辨率过低

**Q: Trust权重如何调整？**
A: 建议alpha=beta=0.5，可通过ablation找最优值

**Q: 如果计算太慢怎么办？**
A: 减少mc_samples，使用更小的数据子集

## 下一步

1. 跑通单个样本的完整流程
2. 选择1-2个UCR数据集
3. 运行deletion实验对比
4. 生成论文图表

## 引用

如果使用本代码，请引用：

```
@article{your_paper,
  title={Multi-view Trust-aware Attribution for Time Series Classification},
  author={Your Name},
  year={2025}
}
```

## License

MIT License