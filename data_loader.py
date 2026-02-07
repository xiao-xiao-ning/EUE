"""
Data Loader - 数据加载工具
支持UCR数据集、自定义CSV、numpy文件等
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Optional
import warnings


class UCRDataLoader:
    """
    UCR时间序列分类数据集加载器
    
    使用方法：
    1. 下载UCR Archive: 
       https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
       
    2. 解压到 data/raw/UCR/
    
    3. 使用:
       loader = UCRDataLoader('data/raw/UCR')
       X_train, y_train, X_test, y_test = loader.load_dataset('ECG200')
    """
    
    def __init__(self, ucr_path: str = 'data/raw/UCR'):
        self.ucr_path = Path(ucr_path)
        
        if not self.ucr_path.exists():
            warnings.warn(
                f"UCR路径不存在: {ucr_path}\n"
                f"请从 https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ 下载"
            )
    
    def list_available_datasets(self):
        """列出可用的数据集"""
        if not self.ucr_path.exists():
            return []
        
        datasets = []
        for item in self.ucr_path.iterdir():
            if item.is_dir():
                train_file = item / f"{item.name}_TRAIN.tsv"
                test_file = item / f"{item.name}_TEST.tsv"
                if train_file.exists() and test_file.exists():
                    datasets.append(item.name)
        
        return sorted(datasets)
    
    def load_dataset(
        self, 
        dataset_name: str,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载UCR数据集
        
        Args:
            dataset_name: 数据集名称（如'ECG200', 'GunPoint'）
            normalize: 是否z-score归一化
            
        Returns:
            X_train: [n_train, 1, length]
            y_train: [n_train]
            X_test: [n_test, 1, length]
            y_test: [n_test]
        """
        dataset_path = self.ucr_path / dataset_name
        
        if not dataset_path.exists():
            raise ValueError(
                f"数据集 {dataset_name} 不存在\n"
                f"可用数据集: {self.list_available_datasets()}"
            )
        
        # 加载训练集
        train_file = dataset_path / f"{dataset_name}_TRAIN.tsv"
        train_data = np.loadtxt(train_file)
        
        y_train = train_data[:, 0].astype(int)
        X_train = train_data[:, 1:]
        
        # 加载测试集
        test_file = dataset_path / f"{dataset_name}_TEST.tsv"
        test_data = np.loadtxt(test_file)
        
        y_test = test_data[:, 0].astype(int)
        X_test = test_data[:, 1:]
        
        # 转换标签为0-based
        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y_train = np.array([label_map[y] for y in y_train])
        y_test = np.array([label_map[y] for y in y_test])
        
        # 添加channel维度
        X_train = X_train[:, np.newaxis, :]  # [n, 1, length]
        X_test = X_test[:, np.newaxis, :]
        
        # 归一化
        if normalize:
            X_train = self._normalize(X_train)
            X_test = self._normalize(X_test)
        
        print(f"加载数据集: {dataset_name}")
        print(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
        print(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")
        print(f"  类别数: {len(unique_labels)}")
        
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        """Z-score归一化"""
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True)
        return (X - mean) / (std + 1e-8)
    
    def create_dataloaders(
        self,
        dataset_name: str,
        batch_size: int = 32,
        normalize: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        创建PyTorch DataLoader
        
        Returns:
            train_loader, test_loader
        """
        X_train, y_train, X_test, y_test = self.load_dataset(
            dataset_name, normalize
        )
        
        # 转换为tensor
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)
        
        # 创建dataset
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        # 创建dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Windows上建议设为0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # 测试时用batch=1方便分析单样本
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, test_loader


class CustomDataLoader:
    """
    自定义数据加载器
    支持多种格式：numpy, CSV, pickle等
    """
    
    @staticmethod
    def load_from_numpy(
        train_path: str,
        test_path: str,
        label_train_path: Optional[str] = None,
        label_test_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从numpy文件加载
        
        假设文件结构：
        - X_train.npy: [n_samples, length] 或 [n_samples, channels, length]
        - y_train.npy: [n_samples]
        
        或者合并在一起：
        - train.npy: [n_samples, length+1] (最后一列是标签)
        """
        X_train = np.load(train_path)
        X_test = np.load(test_path)
        
        # 如果标签分开存储
        if label_train_path and label_test_path:
            y_train = np.load(label_train_path)
            y_test = np.load(label_test_path)
        else:
            # 假设最后一列是标签
            y_train = X_train[:, -1].astype(int)
            y_test = X_test[:, -1].astype(int)
            X_train = X_train[:, :-1]
            X_test = X_test[:, :-1]
        
        # 确保shape正确
        if X_train.ndim == 2:
            X_train = X_train[:, np.newaxis, :]
            X_test = X_test[:, np.newaxis, :]
        
        print(f"从numpy加载数据:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def load_from_csv(
        csv_path: str,
        label_column: str = 'label',
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从CSV加载
        
        CSV格式示例:
        label, t0, t1, t2, ..., tn
        0,     1.2, 1.3, 1.4, ..., 1.5
        1,     2.1, 2.2, 2.3, ..., 2.4
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        # 提取标签
        if label_column in df.columns:
            y = df[label_column].values.astype(int)
            X = df.drop(columns=[label_column]).values
        else:
            # 假设第一列是标签
            y = df.iloc[:, 0].values.astype(int)
            X = df.iloc[:, 1:].values
        
        # 添加channel维度
        X = X[:, np.newaxis, :]
        
        # 归一化
        if normalize:
            mean = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True)
            X = (X - mean) / (std + 1e-8)
        
        print(f"从CSV加载数据: {csv_path}")
        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")
        
        return X, y
    
    @staticmethod
    def save_to_numpy(
        X_train, y_train, X_test, y_test,
        output_dir: str = 'data/processed'
    ):
        """保存为numpy格式"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / 'X_train.npy', X_train)
        np.save(output_path / 'y_train.npy', y_train)
        np.save(output_path / 'X_test.npy', X_test)
        np.save(output_path / 'y_test.npy', y_test)
        
        print(f"数据已保存到: {output_dir}")


# ===== 使用示例 =====

if __name__ == "__main__":
    print("="*60)
    print("数据加载器示例")
    print("="*60)
    
    # ===== 示例1：UCR数据集 =====
    print("\n[示例1] UCR数据集加载")
    print("-"*60)
    
    ucr_loader = UCRDataLoader('data/raw/UCR')
    
    # 列出可用数据集
    available = ucr_loader.list_available_datasets()
    if available:
        print(f"可用数据集 ({len(available)}个):")
        print(", ".join(available[:10]) + "...")
        
        # 加载一个数据集
        # X_train, y_train, X_test, y_test = ucr_loader.load_dataset('ECG200')
        
        # 或直接创建DataLoader
        # train_loader, test_loader = ucr_loader.create_dataloaders(
        #     'ECG200', batch_size=32
        # )
    else:
        print("未找到UCR数据集")
        print("请从 https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ 下载")
    
    # ===== 示例2：自定义numpy数据 =====
    print("\n[示例2] 自定义numpy数据")
    print("-"*60)
    
    # 创建示例数据
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 模拟数据
        X_train = np.random.randn(100, 1, 128)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 1, 128)
        y_test = np.random.randint(0, 2, 20)
        
        # 保存
        CustomDataLoader.save_to_numpy(
            X_train, y_train, X_test, y_test,
            output_dir=tmpdir
        )
        
        # 加载
        X_train_loaded, y_train_loaded, X_test_loaded, y_test_loaded = \
            CustomDataLoader.load_from_numpy(
                os.path.join(tmpdir, 'X_train.npy'),
                os.path.join(tmpdir, 'X_test.npy'),
                os.path.join(tmpdir, 'y_train.npy'),
                os.path.join(tmpdir, 'y_test.npy')
            )
        
        print(f"✓ 加载成功: {X_train_loaded.shape}")
    
    # ===== 示例3：CSV数据 =====
    print("\n[示例3] CSV数据")
    print("-"*60)
    print("CSV格式示例:")
    print("  label, t0, t1, t2, ..., tn")
    print("  0,     1.2, 1.3, 1.4, ..., 1.5")
    print("使用: CustomDataLoader.load_from_csv('your_data.csv')")
    
    print("\n" + "="*60)
    print("数据加载器就绪！")
    print("="*60)