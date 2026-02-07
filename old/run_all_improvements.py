"""
Run All Improvements - 使用所有改进的完整脚本

集成：
1. ✓ Bug修复
2. ✓ Transformer支持
3. ✓ 训练评估分离
4. ✓ Time-step-level consistency
5. ✓ Importance vs Reliability区分
6. ✓ Trust稳定性实验
"""

import torch
import numpy as np
from pathlib import Path
import argparse


def run_complete_workflow_with_improvements():
    """
    完整工作流程（包含所有改进）
    """
    print("="*70)
    print("Multi-view Trust-aware Attribution - 完整改进版")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # ==================== 步骤1：准备数据 ====================
    print("\n" + "="*70)
    print("步骤1：准备数据")
    print("="*70)
    
    # 选项：使用合成数据演示
    from torch.utils.data import DataLoader, TensorDataset
    
    def create_synthetic_data(n_samples=100, length=128):
        X = []
        y = []
        
        for i in range(n_samples):
            t = np.linspace(0, 10, length)
            
            if i < n_samples // 2:
                # Class 0: 低频信号
                signal = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(length)
                label = 0
            else:
                # Class 1: 高频信号
                signal = np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(length)
                label = 1
            
            X.append(signal)
            y.append(label)
        
        X = np.array(X)[:, np.newaxis, :]
        y = np.array(y)
        
        return torch.FloatTensor(X), torch.LongTensor(y)
    
    X_train, y_train = create_synthetic_data(100, 128)
    X_test, y_test = create_synthetic_data(20, 128)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    # ==================== 步骤2：选择模型 ====================
    print("\n" + "="*70)
    print("步骤2：选择模型架构")
    print("="*70)
    
    print("\n可用模型:")
    print("  1. SimpleResNet (默认)")
    print("  2. Transformer")
    print("  3. DeepResNet")
    
    model_choice = input("\n选择模型 (1/2/3，直接回车使用默认): ").strip() or '1'
    
    if model_choice == '1':
        from model_loader import SimpleResNet
        model = SimpleResNet(input_channels=1, num_classes=2, length=128)
        model_name = 'SimpleResNet'
    elif model_choice == '2':
        from other_models import TSTransformer
        model = TSTransformer(
            input_channels=1,
            num_classes=2,
            length=128,
            d_model=128,
            nhead=8,
            num_layers=3
        )
        model_name = 'Transformer'
    elif model_choice == '3':
        from other_models import DeepResNet
        model = DeepResNet(
            input_channels=1,
            num_classes=2,
            num_blocks=[2, 2, 2, 2]
        )
        model_name = 'DeepResNet'
    else:
        print("无效选择，使用默认SimpleResNet")
        from model_loader import SimpleResNet
        model = SimpleResNet(input_channels=1, num_classes=2, length=128)
        model_name = 'SimpleResNet'
    
    print(f"\n选择的模型: {model_name}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== 步骤3：训练模型（分离） ====================
    print("\n" + "="*70)
    print("步骤3：训练模型（独立训练模块）")
    print("="*70)
    
    from separate_train_eval import ModelTrainer
    
    trainer = ModelTrainer(
        model,
        device=device,
        lr=0.001,
        save_dir='./trained_models_improved'
    )
    
    print("\n开始训练...")
    history = trainer.train(
        train_loader,
        test_loader,
        epochs=20,
        model_name=f'{model_name}_demo',
        early_stopping_patience=5,
        verbose=True
    )
    
    # 保存训练日志
    trainer.save_training_log(f'{model_name}_demo')
    
    print(f"\n✓ 模型训练完成")
    print(f"  最佳准确率: {history['best_acc']:.2f}%")
    print(f"  模型保存在: ./trained_models_improved/{model_name}_demo_best.pth")
    
    # ==================== 步骤4：评估解释（分离） ====================
    print("\n" + "="*70)
    print("步骤4：评估解释（独立评估模块）")
    print("="*70)
    
    from separate_train_eval import ExplanationEvaluator
    
    evaluator = ExplanationEvaluator(
        model_path=f'./trained_models_improved/{model_name}_demo_best.pth',
        device=device,
        output_dir='./evaluation_results_improved'
    )
    
    # 重建模型架构并加载权重
    if model_choice == '1':
        from model_loader import SimpleResNet
        eval_model = SimpleResNet(input_channels=1, num_classes=2, length=128)
    elif model_choice == '2':
        from other_models import TSTransformer
        eval_model = TSTransformer(
            input_channels=1, num_classes=2, length=128,
            d_model=128, nhead=8, num_layers=3
        )
    else:
        from other_models import DeepResNet
        eval_model = DeepResNet(
            input_channels=1, num_classes=2, num_blocks=[2, 2, 2, 2]
        )
    
    evaluator.load_model(eval_model)
    
    # ==================== 步骤5：运行改进版实验 ====================
    print("\n" + "="*70)
    print("步骤5：运行改进版实验")
    print("="*70)
    
    from old.complete_improved_pipeline import CompleteExperimentPipeline
    
    exp_pipeline = CompleteExperimentPipeline(
        evaluator.model,
        device=device,
        output_dir='./improved_experiment_results'
    )
    
    # 选择几个测试样本
    n_samples_to_analyze = min(3, len(test_dataset))
    
    for idx in range(n_samples_to_analyze):
        x, y = test_dataset[idx]
        x = x.unsqueeze(0)
        
        print(f"\n分析样本 {idx+1}/{n_samples_to_analyze}")
        print(f"真实标签: {y}")
        
        # 运行所有实验
        exp_pipeline.run_all_experiments(
            x, y,
            sample_name=f'{model_name}_sample_{idx}'
        )
    
    # ==================== 步骤6：总结 ====================
    print("\n" + "="*70)
    print("完整工作流程完成！")
    print("="*70)
    
    print("\n结果保存位置:")
    print(f"  训练模型: ./trained_models_improved/")
    print(f"  训练日志: ./trained_models_improved/{model_name}_demo_training_log.json")
    print(f"  实验结果: ./improved_experiment_results/")
    
    print("\n关键改进:")
    print("  ✓ 训练与评估完全分离")
    print("  ✓ 支持Transformer等多种模型")
    print("  ✓ Time-step-level consistency（不是global）")
    print("  ✓ 明确区分importance和reliability")
    print("  ✓ Trust稳定性实验验证")
    
    print("\n论文写作建议:")
    print("  - 实验1: 放在 4.2 Explanation Uncertainty Analysis")
    print("  - 实验2: 放在 4.3 Cross-view Consistency Evaluation")
    print("  - 实验3: 放在 4.4 Trust-aware Explanation Evaluation")
    print("  - 实验4: 放在 4.4 Trust-aware Explanation Evaluation")
    
    print("\n下一步:")
    print("  1. 在UCR数据集上运行相同流程")
    print("  2. 收集多个数据集的结果")
    print("  3. 生成论文图表")


# ==================== 命令行接口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='运行改进版Multi-view Attribution实验'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='demo',
        choices=['demo', 'train', 'eval'],
        help='运行模式：demo（完整流程），train（仅训练），eval（仅评估）'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='模型路径（eval模式需要）'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        run_complete_workflow_with_improvements()
    
    elif args.mode == 'train':
        print("训练模式...")
        # TODO: 实现独立训练脚本
        
    elif args.mode == 'eval':
        if not args.model_path:
            print("错误：eval模式需要提供 --model-path")
            return
        print(f"评估模式，加载模型: {args.model_path}")
        # TODO: 实现独立评估脚本


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 交互式运行
        run_complete_workflow_with_improvements()
    else:
        # 命令行模式
        main()
