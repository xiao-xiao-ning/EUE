"""Run a simple CNN on the synthetic dataset and visualize attribution, uncertainty, and trust."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from synthetic_data import CausalSpuriousTimeSeriesDataset
from unified_framework import MCDropoutWrapper, UnifiedMultiViewPipeline


def build_dataloaders(batch_size: int = 64, train_ratio: float = 0.8):
    # 强化伪相关段信号，让模型更容易记住它，从而在后续Trust中被判定为不可靠
    dataset = CausalSpuriousTimeSeriesDataset(
        causal_strength=1.5,
        spurious_strength=3.0,
        spurious_flip_prob=0.05
    )
    train_len = int(len(dataset) * train_ratio)
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return dataset, train_loader, test_loader


class SimpleCNN(nn.Module):
    def __init__(self, length: int, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_model(model: nn.Module, train_loader: DataLoader, device: str = "cpu", epochs: int = 5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    #model.eval()
    return model


def visualize_results(timesteps, attribution, uncertainty, trust, consistency, causal_range, spurious_range):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(timesteps, attribution, label="Attribution", linewidth=2)
    ax1.plot(timesteps, trust, label="Trust", linewidth=2)
    ax1.plot(timesteps, consistency, label="Consistency", linewidth=2, linestyle=":")
    ax1.plot(
        timesteps,
        uncertainty,
        label="Uncertainty",
        linewidth=2,
        color="tab:orange",
        linestyle="--"
    )
    ax1.axvspan(causal_range[0], causal_range[1], color="green", alpha=0.1, label="Causal Range")
    ax1.axvspan(spurious_range[0], spurious_range[1], color="red", alpha=0.1, label="Spurious Range")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Attribution / Trust / Consistency / Uncertainty")
    ax1.legend()

    ax1.grid(alpha=0.3)
    plt.title("Attribution vs Uncertainty vs Trust")
    plt.tight_layout()
    plt.savefig("unified_results/synthetic_cnn_results.png")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset, train_loader, test_loader = build_dataloaders()

    model = SimpleCNN(length=dataset.length, num_classes=2).to(device)
    model = train_model(model, train_loader, device=device, epochs=5)

    sample_x, sample_y = next(iter(test_loader))
    sample_x = sample_x.to(device)
    sample_y = sample_y.item()

    pipeline = UnifiedMultiViewPipeline(
        model,
        device=device,
        mc_samples=20,
        trust_epsilon=0.08,
        importance_threshold=0.2
    )
    results = pipeline.compute_complete_explanation(
        sample_x,
        sample_y,
        compute_trust=True,
        trust_n_perturbations=10,
        trust_method='aggregated'
    )
    mapped = {
        k: (v - v.mean()) / (v.std() + 1e-8)
        for k, v in results['mapped_attributions'].items()
    }
    consistency = pipeline.consistency_calculator.compute_timestep_consistency(
        mapped,
        method='inverse_cv'
    )
    results['consistency'] = consistency
    timesteps = np.arange(len(results["attribution_mean"]))
    visualize_results(
        timesteps,
        results["attribution_mean"],
        results["attribution_std"],
        results["trust"],
        results["consistency"],
        dataset.causal_range,
        dataset.spurious_range
    )


if __name__ == "__main__":
    main()
