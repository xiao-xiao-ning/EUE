"""
Separate Training and Evaluation Pipeline
训练与评估分离模块

将模型训练和解释评估完全解耦
支持：
1. 独立训练脚本
2. 独立评估脚本
3. 中间结果保存和加载
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==================== 训练模块 ====================

class ModelTrainer:
    """
    独立的模型训练器
    专注于模型训练，不涉及解释
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        lr: float = 0.001,
        save_dir: str = './trained_models'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'best_acc': 0,
            'best_epoch': 0
        }
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc='Training', leave=False):
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
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 50,
        model_name: str = 'model',
        early_stopping_patience: int = 10,
        verbose: bool = True
    ):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据
            test_loader: 测试数据
            epochs: 训练轮数
            model_name: 模型名称
            early_stopping_patience: 早停耐心值
            verbose: 是否打印信息
        """
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 评估
            test_acc = self.evaluate(test_loader)
            
            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)
            
            # 打印
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Test Acc: {test_acc:.2f}%")
            
            # 保存最佳模型
            if test_acc > self.history['best_acc']:
                self.history['best_acc'] = test_acc
                self.history['best_epoch'] = epoch
                patience_counter = 0
                
                # 保存
                self.save_checkpoint(model_name, epoch, test_acc)
                
                if verbose:
                    print(f"  → 保存最佳模型 (Acc: {test_acc:.2f}%)")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发 (epoch {epoch+1})")
                break
        
        print(f"\n训练完成！")
        print(f"最佳测试准确率: {self.history['best_acc']:.2f}% (Epoch {self.history['best_epoch']+1})")
        
        return self.history
    
    def save_checkpoint(self, model_name: str, epoch: int, accuracy: float):
        """保存checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_name': model_name,
            'model_class': self.model.__class__.__name__,
            'epoch': epoch,
            'accuracy': accuracy,
            'history': self.history
        }
        
        save_path = self.save_dir / f'{model_name}_best.pth'
        torch.save(checkpoint, save_path)
    
    def save_training_log(self, model_name: str):
        """保存训练日志"""
        log_path = self.save_dir / f'{model_name}_training_log.json'
        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=2)


# ==================== 评估模块 ====================

class ExplanationEvaluator:
    """
    独立的解释评估器
    只负责加载已训练模型并进行解释分析
    """
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        output_dir: str = './evaluation_results'
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.checkpoint = torch.load(model_path, map_location=device)
        
        # 需要外部提供模型架构来重建模型
        # 这里先保存checkpoint，等待load_model调用
        self.model = None
        self.model_loaded = False
    
    def load_model(self, model: nn.Module):
        """
        加载模型权重到提供的模型架构
        
        Args:
            model: 已初始化的模型架构
        """
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        self.model_loaded = True
        
        print(f"✓ 模型加载成功")
        print(f"  模型类: {self.checkpoint.get('model_class', 'Unknown')}")
        print(f"  训练准确率: {self.checkpoint.get('accuracy', 0):.2f}%")
        print(f"  训练轮数: {self.checkpoint.get('epoch', 0)}")
    
    def run_explanation_analysis(
        self,
        test_loader: DataLoader,
        pipeline,  # MultiViewAttributionPipeline
        n_samples: int = 10,
        save_name: str = 'explanation_results'
    ):
        """
        运行解释分析
        
        Args:
            test_loader: 测试数据
            pipeline: 已初始化的MultiViewAttributionPipeline
            n_samples: 分析样本数
            save_name: 保存文件名
        """
        if not self.model_loaded:
            raise RuntimeError("请先调用 load_model() 加载模型")
        
        results = []
        
        print(f"\n运行解释分析 ({n_samples} 个样本)...")
        
        for idx, (x, y) in enumerate(test_loader):
            if idx >= n_samples:
                break
            
            x = x.to(self.device)
            y = y.item()
            
            # 计算解释
            explanation = pipeline.compute_multiview_attribution(
                x, y, compute_uncertainty=True
            )
            
            # 保存结果
            results.append({
                'sample_idx': idx,
                'true_label': int(y),
                'consistency': float(explanation['consistency']),
                'mean_trust': float(explanation['trust_score'].mean()),
                'std_trust': float(explanation['trust_score'].std()),
                'mean_uncertainty': float(explanation['attributions']['original']['std'].mean())
            })
            
            print(f"  Sample {idx+1}/{n_samples} - "
                  f"Consistency: {explanation['consistency']:.4f}, "
                  f"Trust: {explanation['trust_score'].mean():.4f}")
        
        # 保存结果
        save_path = self.output_dir / f'{save_name}.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ 解释分析完成，结果保存至: {save_path}")
        
        return results
    
    def run_deletion_experiment(
        self,
        test_loader: DataLoader,
        pipeline,
        deletion_exp,  # DeletionExperiment
        n_samples: int = 30,
        save_name: str = 'deletion_results'
    ):
        """
        运行deletion实验
        
        Args:
            test_loader: 测试数据
            pipeline: MultiViewAttributionPipeline
            deletion_exp: DeletionExperiment
            n_samples: 样本数
            save_name: 保存文件名
        """
        if not self.model_loaded:
            raise RuntimeError("请先调用 load_model() 加载模型")
        
        from old.multiview_attribution import TrustScore
        
        auc_scores = {
            'original': [],
            'trust_weighted': [],
            'random': []
        }
        
        print(f"\n运行Deletion实验 ({n_samples} 个样本)...")
        
        for idx, (x, y) in enumerate(tqdm(test_loader, total=n_samples)):
            if idx >= n_samples:
                break
            
            x = x.to(self.device)
            y = y.item()
            
            # 计算解释
            explanation = pipeline.compute_multiview_attribution(
                x, y, compute_uncertainty=True
            )
            
            mean_attr = explanation['attributions']['original']['mean']
            trust_score = explanation['trust_score']
            trusted_attr = TrustScore.get_trusted_attribution(mean_attr, trust_score)
            random_attr = np.random.randn(len(mean_attr))
            
            # Deletion曲线
            attributions = {
                'Original': mean_attr,
                'Trust-weighted': trusted_attr,
                'Random': random_attr
            }
            
            results = deletion_exp.compare_attributions(
                x, y, attributions, mode='deletion'
            )
            
            # 计算AUC
            for method, (fracs, scores) in results.items():
                auc = deletion_exp.compute_auc(fracs, scores)
                
                if method == 'Original':
                    auc_scores['original'].append(auc)
                elif method == 'Trust-weighted':
                    auc_scores['trust_weighted'].append(auc)
                elif method == 'Random':
                    auc_scores['random'].append(auc)
        
        # 统计结果
        summary = {
            method: {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'median': float(np.median(scores))
            }
            for method, scores in auc_scores.items()
        }
        
        # 保存
        save_path = self.output_dir / f'{save_name}.json'
        with open(save_path, 'w') as f:
            json.dump({
                'summary': summary,
                'raw_scores': {k: [float(v) for v in vals] 
                              for k, vals in auc_scores.items()}
            }, f, indent=2)
        
        print(f"\n✓ Deletion实验完成，结果保存至: {save_path}")
        print("\n结果摘要:")
        for method, stats in summary.items():
            print(f"  {method}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return summary


# ==================== 使用示例 ====================

def example_training_workflow():
    """示例：训练流程"""
    print("="*60)
    print("示例：独立训练流程")
    print("="*60)
    
    # 1. 准备数据
    from torch.utils.data import TensorDataset, DataLoader
    
    X_train = torch.randn(100, 1, 128)
    y_train = torch.randint(0, 2, (100,))
    X_test = torch.randn(20, 1, 128)
    y_test = torch.randint(0, 2, (20,))
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=16, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=1
    )
    
    # 2. 创建模型
    from model_loader import SimpleResNet
    model = SimpleResNet(input_channels=1, num_classes=2, length=128)
    
    # 3. 训练
    trainer = ModelTrainer(
        model,
        device='cuda',
        save_dir='./trained_models'
    )
    
    history = trainer.train(
        train_loader,
        test_loader,
        epochs=20,
        model_name='my_model',
        verbose=True
    )
    
    # 4. 保存训练日志
    trainer.save_training_log('my_model')
    
    print("\n训练完成！模型保存至: ./trained_models/my_model_best.pth")


def example_evaluation_workflow():
    """示例：评估流程"""
    print("="*60)
    print("示例：独立评估流程")
    print("="*60)
    
    # 1. 准备测试数据
    from torch.utils.data import TensorDataset, DataLoader
    
    X_test = torch.randn(20, 1, 128)
    y_test = torch.randint(0, 2, (20,))
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=1
    )
    
    # 2. 创建评估器
    evaluator = ExplanationEvaluator(
        model_path='./trained_models/my_model_best.pth',
        device='cuda',
        output_dir='./evaluation_results'
    )
    
    # 3. 加载模型架构
    from model_loader import SimpleResNet
    model = SimpleResNet(input_channels=1, num_classes=2, length=128)
    evaluator.load_model(model)
    
    # 4. 运行解释分析
    from old.multiview_attribution import MultiViewAttributionPipeline
    pipeline = MultiViewAttributionPipeline(
        evaluator.model,
        device='cuda',
        mc_samples=20
    )
    
    results = evaluator.run_explanation_analysis(
        test_loader,
        pipeline,
        n_samples=5,
        save_name='explanation_results'
    )
    
    # 5. 运行deletion实验
    from deletion_experiment import DeletionExperiment
    deletion_exp = DeletionExperiment(evaluator.model, 'cuda')
    
    deletion_results = evaluator.run_deletion_experiment(
        test_loader,
        pipeline,
        deletion_exp,
        n_samples=10,
        save_name='deletion_results'
    )
    
    print("\n评估完成！结果保存至: ./evaluation_results/")


if __name__ == "__main__":
    print("""
    训练与评估分离模块使用说明
    ========================
    
    步骤1：独立训练模型
    -------------------
    python train_model.py
    
    步骤2：独立评估解释
    -------------------
    python evaluate_explanations.py
    
    或使用：
    >>> example_training_workflow()    # 训练
    >>> example_evaluation_workflow()  # 评估
    """)
