import argparse
import torch
import numpy as np
from model_loader import SimpleResNet  # 或你想用的模型
from unified_experiments import UnifiedExperimentPipeline
from data_loader import UCRDataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run unified experiments on a specified UCR dataset.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='CinCECGTorso',
        help='Name of the UCR dataset to load (default: CinCECGTorso)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ucr_loader = UCRDataLoader('data/raw/UCR')
    X_train, y_train, X_test, y_test = ucr_loader.load_dataset(dataset_name)

    # UCR加载默认返回float64，显式转换成float32以匹配模型参数dtype
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    x = torch.from_numpy(X_test[0]).unsqueeze(0).to(device)
    y = int(y_test[0])
    num_classes = len(np.unique(y_train))
    series_length = X_train.shape[-1]

    model = SimpleResNet(
        input_channels=X_train.shape[1],
        num_classes=num_classes,
        length=series_length
    ).to(device)
    model.eval()
    # 如果有预训练权重，就在这里加载
    # model.load_state_dict(torch.load('your_model.pth'))

    # 运行实验
    exp = UnifiedExperimentPipeline(model, device=device)
    exp.run_all_experiments(x, y, name=f'{dataset_name}_sample')

    print("结果已写入 ./unified_results/")


if __name__ == '__main__':
    main()