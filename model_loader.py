"""
Model Loader - 模型加载工具
包含常用的时间序列分类模型架构和加载函数
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict


# ===== 模型架构定义 =====

class SimpleResNet(nn.Module):
    """
    简单的ResNet用于时间序列分类
    适用于大多数UCR数据集
    """
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        length: int = 128,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 2)
        
        # 残差连接
        self.residual = nn.Conv1d(input_channels, hidden_dim * 2, kernel_size=1)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: [batch, channels, length]
        identity = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 残差连接
        out = out + identity
        out = self.relu(out)
        
        out = self.global_pool(out)
        out = out.squeeze(-1)
        out = self.fc(out)
        
        return out


class FCN(nn.Module):
    """
    Fully Convolutional Network
    经典的时间序列分类模型
    """
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        length: int = 128
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        
        out = self.global_pool(out)
        out = out.squeeze(-1)
        out = self.fc(out)
        
        return out


class InceptionTime(nn.Module):
    """
    InceptionTime模型（简化版）
    """
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        length: int = 128,
        n_filters: int = 32
    ):
        super().__init__()
        
        self.inception1 = self._make_inception_block(input_channels, n_filters)
        self.inception2 = self._make_inception_block(n_filters * 4, n_filters)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters * 4, num_classes)
    
    def _make_inception_block(self, in_channels, n_filters):
        # 不同kernel size的并行卷积
        conv1 = nn.Conv1d(in_channels, n_filters, kernel_size=1)
        conv3 = nn.Conv1d(in_channels, n_filters, kernel_size=3, padding=1)
        conv5 = nn.Conv1d(in_channels, n_filters, kernel_size=5, padding=2)
        maxpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1)
        )
        
        return nn.ModuleList([conv1, conv3, conv5, maxpool])
    
    def forward(self, x):
        # Inception block 1
        out1 = torch.cat([conv(x) for conv in self.inception1], dim=1)
        out1 = torch.relu(out1)
        
        # Inception block 2
        out2 = torch.cat([conv(out1) for conv in self.inception2], dim=1)
        out2 = torch.relu(out2)
        
        out = self.global_pool(out2)
        out = out.squeeze(-1)
        out = self.fc(out)
        
        return out


class LSTM_FCN(nn.Module):
    """
    LSTM-FCN混合模型
    结合LSTM和FCN的优势
    """
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        length: int = 128,
        lstm_hidden: int = 64,
        num_layers: int = 1
    ):
        super().__init__()
        
        # LSTM分支
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # FCN分支
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 合并后的全连接层
        self.fc = nn.Linear(lstm_hidden + 128, num_classes)
    
    def forward(self, x):
        # LSTM分支: [batch, channels, length] -> [batch, length, channels]
        lstm_in = x.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步
        
        # FCN分支
        fcn_out = self.conv1(x)
        fcn_out = self.bn1(fcn_out)
        fcn_out = self.relu(fcn_out)
        
        fcn_out = self.conv2(fcn_out)
        fcn_out = self.bn2(fcn_out)
        fcn_out = self.relu(fcn_out)
        
        fcn_out = self.conv3(fcn_out)
        fcn_out = self.bn3(fcn_out)
        fcn_out = self.relu(fcn_out)
        
        fcn_out = self.global_pool(fcn_out).squeeze(-1)
        
        # 合并
        combined = torch.cat([lstm_out, fcn_out], dim=1)
        out = self.fc(combined)
        
        return out


# ===== 模型加载/保存函数 =====

def save_model(
    model: nn.Module,
    save_path: str,
    config: Optional[Dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None
):
    """
    保存模型
    
    Args:
        model: PyTorch模型
        save_path: 保存路径（如'models/checkpoints/my_model.pth'）
        config: 模型配置字典
        optimizer: 优化器（可选）
        epoch: 训练轮数（可选）
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if config is not None:
        save_dict['config'] = config
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        save_dict['epoch'] = epoch
    
    # 确保目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(save_dict, save_path)
    print(f"模型已保存到: {save_path}")


def load_model(
    model_class: type,
    checkpoint_path: str,
    device: str = 'cuda',
    strict: bool = True
) -> nn.Module:
    """
    加载模型
    
    Args:
        model_class: 模型类（如SimpleResNet）
        checkpoint_path: checkpoint路径
        device: 设备
        strict: 是否严格匹配state_dict
        
    Returns:
        加载好的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 如果保存了config，用config初始化模型
    if 'config' in checkpoint:
        model = model_class(**checkpoint['config'])
    else:
        # 否则用默认参数初始化
        print("警告: checkpoint中没有config，使用默认参数初始化模型")
        model = model_class()
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    model = model.to(device)
    model.eval()
    
    print(f"模型已从 {checkpoint_path} 加载")
    if 'epoch' in checkpoint:
        print(f"  训练轮数: {checkpoint['epoch']}")
    
    return model


def load_pretrained_model(
    checkpoint_path: str,
    device: str = 'cuda'
) -> nn.Module:
    """
    加载预训练模型（自动识别模型类型）
    
    Args:
        checkpoint_path: checkpoint路径
        device: 设备
        
    Returns:
        加载好的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' not in checkpoint:
        raise ValueError(
            "checkpoint中必须包含config字段来识别模型类型\n"
            "请使用 save_model() 函数保存模型"
        )
    
    config = checkpoint['config']
    
    # 根据config识别模型类型
    if 'model_type' in config:
        model_type = config['model_type']
        config_copy = config.copy()
        config_copy.pop('model_type')
    else:
        # 默认使用SimpleResNet
        model_type = 'SimpleResNet'
        config_copy = config
    
    # 模型字典
    MODEL_DICT = {
        'SimpleResNet': SimpleResNet,
        'FCN': FCN,
        'InceptionTime': InceptionTime,
        'LSTM_FCN': LSTM_FCN
    }
    
    if model_type not in MODEL_DICT:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    model = MODEL_DICT[model_type](**config_copy)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"已加载预训练模型: {model_type}")
    
    return model


# ===== 模型训练器 =====

class SimpleTrainer:
    """
    简单的模型训练器
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        lr: float = 0.001
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(
        self,
        train_loader,
        test_loader,
        epochs: int = 50,
        save_path: Optional[str] = None,
        verbose: bool = True
    ):
        """完整训练流程"""
        best_acc = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            test_acc = self.evaluate(test_loader)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Test Acc: {test_acc:.2f}%")
            
            # 保存最佳模型
            if save_path and test_acc > best_acc:
                best_acc = test_acc
                save_model(
                    self.model,
                    save_path,
                    config={'model_type': self.model.__class__.__name__},
                    optimizer=self.optimizer,
                    epoch=epoch
                )
        
        print(f"训练完成! 最佳测试准确率: {best_acc:.2f}%")
        return best_acc


# ===== 使用示例 =====

if __name__ == "__main__":
    print("="*60)
    print("模型加载器示例")
    print("="*60)
    
    # ===== 示例1：创建和保存模型 =====
    print("\n[示例1] 创建和保存模型")
    print("-"*60)
    
    model = SimpleResNet(
        input_channels=1,
        num_classes=2,
        length=128,
        hidden_dim=64
    )
    
    print(f"创建模型: {model.__class__.__name__}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 保存
    config = {
        'model_type': 'SimpleResNet',
        'input_channels': 1,
        'num_classes': 2,
        'length': 128,
        'hidden_dim': 64
    }
    
    # save_model(model, 'models/checkpoints/example.pth', config=config)
    
    # ===== 示例2：可用的模型 =====
    print("\n[示例2] 可用的模型架构")
    print("-"*60)
    
    models = {
        'SimpleResNet': SimpleResNet,
        'FCN': FCN,
        'InceptionTime': InceptionTime,
        'LSTM_FCN': LSTM_FCN
    }
    
    for name, model_class in models.items():
        model = model_class(input_channels=1, num_classes=2, length=128)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name:20s} - 参数量: {params:,}")
    
    # ===== 示例3：训练模型 =====
    print("\n[示例3] 训练模型")
    print("-"*60)
    print("使用 SimpleTrainer 类快速训练:")
    print("""
    trainer = SimpleTrainer(model, device='cuda', lr=0.001)
    trainer.train(
        train_loader, 
        test_loader, 
        epochs=50,
        save_path='models/checkpoints/my_model.pth'
    )
    """)
    
    print("\n" + "="*60)
    print("模型加载器就绪！")
    print("="*60)