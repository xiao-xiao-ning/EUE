import torch
from torch.utils.data import random_split
from model_loader import SimpleResNet  # 或你想用的模型
from unified_experiments import UnifiedExperimentPipeline
from synthetic_data import CausalSpuriousTimeSeriesDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 构造合成数据集
dataset = CausalSpuriousTimeSeriesDataset(
    n_samples=1000,
    length=200,
    causal_range=(40, 70),
    spurious_range=(120, 150),
    noise_std=0.5,
    causal_strength=2.0,
    spurious_strength=1.5,
    spurious_flip_prob=0.3,
    seed=42
)

# 2. 简单划分一下数据（可选）
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

# 3. 抽一个测试样本
x, y = test_set[0]
x = x.unsqueeze(0).to(device)   # [1, 1, length]
y = int(y)

# 4. 准备模型（保持输入长度与数据一致）
model = SimpleResNet(
    input_channels=1,
    num_classes=2,
    length=x.shape[-1]
).to(device)
model.eval()
# 如果有预训练权重，就在这里加载
# model.load_state_dict(torch.load('your_model.pth'))

# 5. 运行实验
exp = UnifiedExperimentPipeline(model, device=device)
exp.run_all_experiments(x, y, name='synthetic_sample')

print("结果已写入 ./unified_results/")