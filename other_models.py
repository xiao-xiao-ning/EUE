"""
Enhanced Model Loader - 增强版模型加载器
支持Transformer、更多ResNet变体、以及任意自定义模型
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict


# ===== Transformer模型 =====

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, length, d_model]
        return x + self.pe[:, :x.size(1)]


class TSTransformer(nn.Module):
    """
    时间序列Transformer分类模型
    
    Args:
        input_channels: 输入通道数
        num_classes: 类别数
        d_model: Transformer维度（默认128）
        nhead: 注意力头数（默认8）
        num_layers: Transformer层数（默认3）
        dim_feedforward: FFN维度（默认512）
        dropout: Dropout率（默认0.1）
    """
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        length: int = 128,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 输入投影
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=length)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: [batch, channels, length]
        x = x.transpose(1, 2)  # [batch, length, channels]
        
        # 投影到d_model维度
        x = self.input_projection(x)  # [batch, length, d_model]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # [batch, length, d_model]
        
        # 全局平均池化
        x = x.mean(dim=1)  # [batch, d_model]
        
        # 分类
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# ===== 更深的ResNet =====

class ResidualBlock(nn.Module):
    """ResNet残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class DeepResNet(nn.Module):
    """
    更深的ResNet (ResNet-18/34风格)
    
    Args:
        input_channels: 输入通道数
        num_classes: 类别数
        num_blocks: 每个stage的block数量 [2, 2, 2, 2]
        base_channels: 基础通道数（默认64）
    """
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        length: int = 128,
        num_blocks: list = [2, 2, 2, 2],
        base_channels: int = 64
    ):
        super().__init__()
        
        self.in_channels = base_channels
        
        # 初始卷积
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=7, 
                              stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Stages
        self.stage1 = self._make_stage(base_channels, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(base_channels*2, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(base_channels*4, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(base_channels*8, num_blocks[3], stride=2)
        
        # 全局池化和分类
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels*8, num_classes)
    
    def _make_stage(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        
        return x


# ===== 模型注册表 =====

MODEL_REGISTRY = {
    'SimpleResNet': None,  # 从原来的model_loader导入
    'FCN': None,
    'InceptionTime': None,
    'LSTM_FCN': None,
    'TSTransformer': TSTransformer,
    'DeepResNet': DeepResNet,
}


def register_custom_model(name: str, model_class):
    """
    注册自定义模型
    
    使用方法：
    >>> register_custom_model('MyModel', MyModelClass)
    >>> model = create_model('MyModel', input_channels=1, num_classes=2)
    """
    MODEL_REGISTRY[name] = model_class
    print(f"已注册模型: {name}")


def create_model(
    model_name: str,
    input_channels: int = 1,
    num_classes: int = 2,
    length: int = 128,
    **kwargs
) -> nn.Module:
    """
    创建模型的统一接口
    
    Args:
        model_name: 模型名称
        input_channels: 输入通道数
        num_classes: 类别数
        length: 序列长度
        **kwargs: 模型特定参数
        
    Returns:
        模型实例
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型: {model_name}\n"
            f"可用模型: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    
    # 构建参数
    model_kwargs = {
        'input_channels': input_channels,
        'num_classes': num_classes,
        'length': length
    }
    model_kwargs.update(kwargs)
    
    model = model_class(**model_kwargs)
    
    return model


def load_any_model(
    checkpoint_path: str,
    device: str = 'cuda'
) -> nn.Module:
    """
    加载任意模型（自动识别类型）
    
    checkpoint必须包含:
    - model_state_dict
    - model_name (或 model_type)
    - model_config
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    elif 'model_type' in checkpoint:
        model_name = checkpoint['model_type']
    else:
        raise ValueError("checkpoint必须包含 model_name 或 model_type")
    
    config = checkpoint.get('model_config', {})
    
    model = create_model(model_name, **config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"已加载模型: {model_name}")
    return model


# ===== 使用示例 =====

if __name__ == "__main__":
    print("="*60)
    print("增强版模型加载器")
    print("="*60)
    
    # 示例1：使用Transformer
    print("\n[示例1] Transformer模型")
    transformer = TSTransformer(
        input_channels=1,
        num_classes=2,
        d_model=128,
        nhead=8,
        num_layers=3
    )
    print(f"参数量: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # 示例2：使用DeepResNet
    print("\n[示例2] DeepResNet模型")
    deep_resnet = DeepResNet(
        input_channels=1,
        num_classes=2,
        num_blocks=[2, 2, 2, 2]
    )
    print(f"参数量: {sum(p.numel() for p in deep_resnet.parameters()):,}")
    
    # 示例3：注册自定义模型
    print("\n[示例3] 注册自定义模型")
    
    class MyCustomModel(nn.Module):
        def __init__(self, input_channels, num_classes, length):
            super().__init__()
            self.conv = nn.Conv1d(input_channels, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(64, num_classes)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x).squeeze(-1)
            return self.fc(x)
    
    register_custom_model('MyModel', MyCustomModel)
    my_model = create_model('MyModel', input_channels=1, num_classes=2)
    print(f"自定义模型参数量: {sum(p.numel() for p in my_model.parameters()):,}")
    
    print("\n" + "="*60)
    print("增强版模型加载器就绪！")
    print("="*60)
