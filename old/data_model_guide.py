"""
数据和模型准备指南
Data and Model Setup Guide

本文件说明如何准备你的数据和模型
"""

# ===== 推荐的项目目录结构 =====

"""
your_project/
│
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   │   ├── ECG200/                 # UCR数据集示例
│   │   │   ├── ECG200_TRAIN.tsv
│   │   │   └── ECG200_TEST.tsv
│   │   └── your_dataset/           # 你自己的数据
│   │
│   └── processed/                  # 预处理后的数据
│       ├── train.pt
│       └── test.pt
│
├── models/                         # 模型目录
│   ├── checkpoints/                # 训练好的模型权重
│   │   ├── resnet_ecg200_best.pth
│   │   └── your_model.pth
│   │
│   └── architectures/              # 模型定义文件
│       ├── resnet.py
│       └── your_model.py
│
├── outputs/                        # 输出目录
│   ├── visualizations/             # 可视化结果
│   ├── experiments/                # 实验结果
│   └── logs/                       # 日志文件
│
├── multiview_attribution.py        # 我们的框架代码
├── deletion_experiment.py
├── visualization.py
├── end_to_end_example.py
├── quick_test.py
│
├── data_loader.py                  # 数据加载器（新建）
├── model_loader.py                 # 模型加载器（新建）
└── run_experiments.py              # 运行实验脚本（新建）
"""


# ===== 方案1：使用UCR数据集（最简单，推荐新手） =====

"""
步骤：
1. 下载UCR Archive
2. 使用我们提供的loader加载
3. 训练或加载预训练模型
"""

# 示例代码见下方 data_loader.py


# ===== 方案2：使用你自己的数据 =====

"""
数据格式要求：
- 时间序列数据：numpy array 或 torch tensor
- Shape: [n_samples, n_channels, length]
  - n_samples: 样本数量
  - n_channels: 通道数（单变量=1，多变量>1）
  - length: 时间步长度
  
- 标签：numpy array 或 torch tensor
- Shape: [n_samples]
  - 每个样本的类别标签（0, 1, 2, ...）

示例：
X_train.shape = (100, 1, 128)  # 100个样本，1通道，长度128
y_train.shape = (100,)          # 100个标签
"""


# ===== 方案3：使用预训练模型 =====

"""
如果你已经有训练好的模型：

1. 保存模型（推荐方式）：
   torch.save({
       'model_state_dict': model.state_dict(),
       'config': {...}  # 模型配置
   }, 'models/checkpoints/your_model.pth')

2. 加载模型：
   checkpoint = torch.load('models/checkpoints/your_model.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()

3. 确保模型输出shape：[batch_size, num_classes]
"""


# ===== 完整工作流程示例 =====

def example_workflow():
    """
    从零开始的完整工作流程
    """
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    
    # ===== 步骤1：准备数据 =====
    print("步骤1：加载数据...")
    
    # 选项A：从文件加载
    # X_train = np.load('data/processed/X_train.npy')
    # y_train = np.load('data/processed/y_train.npy')
    
    # 选项B：从UCR加载（见 data_loader.py）
    # from data_loader import UCRDataLoader
    # loader = UCRDataLoader('data/raw/UCR')
    # X_train, y_train, X_test, y_test = loader.load_dataset('ECG200')
    
    # 选项C：使用你的数据（示例）
    # 假设你有CSV或其他格式
    # import pandas as pd
    # df = pd.read_csv('your_data.csv')
    # X_train = df[['feature1', 'feature2', ...]].values
    # X_train = X_train.reshape(-1, 1, length)  # 转换为正确shape
    
    # 为演示，这里用随机数据
    X_train = np.random.randn(100, 1, 128)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(20, 1, 128)
    y_test = np.random.randint(0, 2, 20)
    
    # 转换为tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    print(f"  数据shape: {X_train.shape}")
    
    # ===== 步骤2：准备模型 =====
    print("\n步骤2：准备模型...")
    
    # 选项A：加载已训练模型
    # from model_loader import load_pretrained_model
    # model = load_pretrained_model('models/checkpoints/your_model.pth')
    
    # 选项B：定义并训练新模型
    from model_loader import SimpleResNet  # 或你自己的模型
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleResNet(
        input_channels=1,
        num_classes=2,
        length=128
    ).to(device)
    
    # 训练模型（简化版）
    print("  训练模型...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(5):  # 简化训练
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    print("  ✓ 模型准备完成")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_channels': 1,
            'num_classes': 2,
            'length': 128
        }
    }, 'models/checkpoints/my_model.pth')
    
    # ===== 步骤3：运行我们的框架 =====
    print("\n步骤3：运行Multi-view Attribution...")
    
    from multiview_attribution import MultiViewAttributionPipeline
    
    pipeline = MultiViewAttributionPipeline(
        model=model,
        device=device,
        mc_samples=20
    )
    
    # 选一个测试样本
    x_sample = X_test_tensor[0:1].to(device)
    y_sample = y_test_tensor[0].item()
    
    results = pipeline.compute_multiview_attribution(
        x_sample, y_sample, compute_uncertainty=True
    )
    
    print(f"  ✓ 一致性: {results['consistency']:.4f}")
    print(f"  ✓ Trust score: {results['trust_score'].mean():.4f}")
    
    # ===== 步骤4：运行实验和可视化 =====
    print("\n步骤4：运行实验...")
    
    from deletion_experiment import DeletionExperiment
    
    exp = DeletionExperiment(model, device)
    
    mean_attr = results['attributions']['original']['mean']
    trusted_attr = results['trust_score'] * mean_attr
    
    deletion_results = exp.compare_attributions(
        x_sample, y_sample,
        {
            'Original': mean_attr,
            'Trust-weighted': trusted_attr
        },
        mode='deletion'
    )
    
    print("  ✓ Deletion实验完成")
    
    return model, test_loader, pipeline


if __name__ == "__main__":
    print("="*60)
    print("数据和模型准备指南")
    print("="*60)
    print("\n请查看本文件顶部的目录结构说明")
    print("\n接下来创建以下辅助文件：")
    print("  1. data_loader.py - 数据加载器")
    print("  2. model_loader.py - 模型加载器")
    print("  3. run_experiments.py - 实验运行脚本")
    print("\n运行 example_workflow() 查看完整流程")