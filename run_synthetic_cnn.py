"""Run a simple CNN on the synthetic dataset and visualize attribution, uncertainty, and trust."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from tqdm import tqdm

from typing import Optional

from synthetic_data import CausalSpuriousTimeSeriesDataset
from unified_framework import UnifiedMultiViewPipeline


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


def perturb_timeseries(x: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
    """Apply small, semantic-neutral Gaussian noise."""
    base = x.std()
    base_val = base.item() if torch.isfinite(base).item() else 0.0
    scale = base_val if base_val > 0 else 1.0
    noise = torch.randn_like(x) * noise_std * scale
    return x + noise


def predict_label(model: nn.Module, x: torch.Tensor) -> tuple[int, float]:
    with torch.no_grad():
        logits = model(x)
    probs = F.softmax(logits, dim=-1)
    conf, pred = torch.max(probs, dim=-1)
    return pred.item(), conf.item()


def explanation_topk_overlap(e0, e1, topk_ratio: float = 0.1) -> float:
    e0 = np.asarray(e0).flatten()
    e1 = np.asarray(e1).flatten()
    length = min(len(e0), len(e1))
    if length == 0:
        return 0.0
    k = max(1, int(length * topk_ratio))
    idx0 = np.argpartition(np.abs(e0), -k)[-k:]
    idx1 = np.argpartition(np.abs(e1), -k)[-k:]
    set0 = set(idx0.tolist())
    set1 = set(idx1.tolist())
    union = len(set0 | set1)
    if union == 0:
        return 0.0
    return len(set0 & set1) / union


def classify_behavior(pred_same: bool, overlap: float, trust_delta: float, overlap_thresh: float = 0.5) -> dict:
    expl_stable = overlap > overlap_thresh
    case_label = (
        'pred_stable_expl_stable' if pred_same and expl_stable else
        'pred_stable_expl_unstable' if pred_same and not expl_stable else
        'pred_unstable_expl_stable' if (not pred_same) and expl_stable else
        'pred_unstable_expl_unstable'
    )
    return {
        'pred_same': pred_same,
        'expl_stable': expl_stable,
        'trust_change': trust_delta,
        'case_label': case_label
    }


def evaluate_with_perturbation(
    pipeline: UnifiedMultiViewPipeline,
    model: nn.Module,
    x: torch.Tensor,
    noise_std: float = 0.01,
    topk_ratio: float = 0.1,
    trust_kwargs: Optional[dict] = None
) -> dict:
    trust_kwargs = trust_kwargs or {}

    y0, conf0 = predict_label(model, x)
    original = pipeline.compute_complete_explanation(
        x,
        y0,
        compute_trust=True,
        **trust_kwargs
    )

    x_pert = perturb_timeseries(x, noise_std=noise_std)
    y1, conf1 = predict_label(model, x_pert)
    perturbed = pipeline.compute_complete_explanation(
        x_pert,
        y1,
        compute_trust=True,
        **trust_kwargs
    )

    overlap = explanation_topk_overlap(
        original['attribution_mean'],
        perturbed['attribution_mean'],
        topk_ratio=topk_ratio
    )
    t0_mean = float(np.mean(original['trust']))
    t1_mean = float(np.mean(perturbed['trust']))

    return {
        'y0': y0,
        'y1': y1,
        'conf0': conf0,
        'conf1': conf1,
        't0_mean': t0_mean,
        't1_mean': t1_mean,
        'trust_delta': t1_mean - t0_mean,
        'overlap': overlap
    }


def run_trust_perturbation_protocol(
    pipeline: UnifiedMultiViewPipeline,
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    max_samples: int = 128,
    noise_std: float = 0.01,
    topk_ratio: float = 0.1,
    overlap_thresh: float = 0.5,
    scatter_path: str = "unified_results/trust_overlap_vs_delta.png",
    case_path: str = "unified_results/trust_case_distribution.png"
) -> dict:
    overlaps, trust_deltas = [], []
    case_storage = {
        'pred_stable_expl_stable': [],
        'pred_stable_expl_unstable': [],
        'pred_unstable_expl_stable': [],
        'pred_unstable_expl_unstable': []
    }

    trust_kwargs = {
        'trust_n_perturbations': 5,
        'trust_method': 'aggregated'
    }

    model.eval()
    processed = 0
    for x, _ in data_loader:
        if processed >= max_samples:
            break
        x = x.to(device)
        eval_result = evaluate_with_perturbation(
            pipeline,
            model,
            x,
            noise_std=noise_std,
            topk_ratio=topk_ratio,
            trust_kwargs=trust_kwargs
        )

        overlaps.append(eval_result['overlap'])
        trust_deltas.append(eval_result['trust_delta'])

        behavior = classify_behavior(
            eval_result['y0'] == eval_result['y1'],
            eval_result['overlap'],
            eval_result['trust_delta'],
            overlap_thresh=overlap_thresh
        )
        case_storage[behavior['case_label']].append(eval_result['t1_mean'])
        processed += 1

    if processed == 0:
        return {
            'mean_overlap': 0.0,
            'mean_trust_delta': 0.0,
            'case_counts': {k: 0 for k in case_storage}
        }

    if overlaps:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(overlaps, trust_deltas, alpha=0.7)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Explanation overlap (top-k)')
        ax.set_ylabel('Trust Δ (perturbed - original)')
        ax.set_title('Trust change vs explanation overlap')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=300)
        plt.close(fig)

    case_labels = {
        'pred_stable_expl_stable': 'pred✓ expl✓',
        'pred_stable_expl_unstable': 'pred✓ expl✗',
        'pred_unstable_expl_stable': 'pred✗ expl✓',
        'pred_unstable_expl_unstable': 'pred✗ expl✗'
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    data = []
    labels = []
    counts = []
    for key in case_storage:
        values = case_storage[key]
        data.append(values if values else [np.nan])
        labels.append(case_labels[key])
        counts.append(len(values))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel('Trust (perturbed sample)')
    ax.set_title('Trust distributions across consistency cases')
    ylim = ax.get_ylim()
    text_y = ylim[1] - 0.05 * (ylim[1] - ylim[0])
    for idx, count in enumerate(counts, start=1):
        ax.text(idx, text_y, f"n={count}", ha='center', va='bottom', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(case_path, dpi=300)
    plt.close(fig)

    return {
        'mean_overlap': float(np.nanmean(overlaps)) if overlaps else 0.0,
        'mean_trust_delta': float(np.nanmean(trust_deltas)) if trust_deltas else 0.0,
        'case_counts': {k: len(v) for k, v in case_storage.items()}
    }


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
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax_top.plot(timesteps, attribution, label="Attribution", linewidth=2)
    ax_top.plot(timesteps, trust, label="Trust", linewidth=2)
    ax_top.plot(timesteps, consistency, label="Consistency", linewidth=2, linestyle=":")
    ax_top.axvspan(causal_range[0], causal_range[1], color="green", alpha=0.1, label="Causal Range")
    ax_top.axvspan(spurious_range[0], spurious_range[1], color="red", alpha=0.1, label="Spurious Range")
    ax_top.set_ylabel("Attribution / Trust / Consistency")
    ax_top.legend(loc="upper right")
    ax_top.grid(alpha=0.3)

    ax_bottom.plot(
        timesteps,
        uncertainty,
        label="Uncertainty",
        linewidth=2,
        color="tab:orange",
        linestyle="--"
    )
    ax_bottom.fill_between(timesteps, 0, uncertainty, color="tab:orange", alpha=0.2)
    ax_bottom.axvspan(causal_range[0], causal_range[1], color="green", alpha=0.1)
    ax_bottom.axvspan(spurious_range[0], spurious_range[1], color="red", alpha=0.1)
    ax_bottom.set_xlabel("Timestep")
    ax_bottom.set_ylabel("Uncertainty")
    ax_bottom.legend(loc="upper right")
    ax_bottom.grid(alpha=0.3)

    fig.suptitle("Attribution vs Uncertainty vs Trust", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig("unified_results/synthetic_cnn_results.png")


def compute_segment_mean(values: np.ndarray, mask: np.ndarray) -> float:
    seg = values[mask]
    if seg.size == 0:
        return 0.0
    return float(seg.mean())


def evaluate_over_batch(
    pipeline: UnifiedMultiViewPipeline,
    data_loader: DataLoader,
    dataset: CausalSpuriousTimeSeriesDataset,
    device: str,
    max_samples: int = 64
) -> dict:
    causal_mask = dataset.get_causal_mask().astype(bool)
    spurious_mask = dataset.get_spurious_mask().astype(bool)

    trust_causal, trust_spurious = [], []
    attr_causal, attr_spurious = [], []
    trust_concat, attr_concat, label_concat = [], [], []

    total_iters = min(len(data_loader), max_samples)
    progress = tqdm(data_loader, total=total_iters, desc="Evaluating batch", leave=False)

    for idx, (x, y) in enumerate(progress):
        if idx >= max_samples:
            break
        x = x.to(device)
        y = y.item()
        results = pipeline.compute_complete_explanation(
            x,
            y,
            compute_trust=True,
            trust_n_perturbations=5,
            trust_method='aggregated'
        )
        trust = results['trust']
        attribution = results['attribution_mean']

        trust_causal.append(compute_segment_mean(trust, causal_mask))
        trust_spurious.append(compute_segment_mean(trust, spurious_mask))
        attr_causal.append(compute_segment_mean(attribution, causal_mask))
        attr_spurious.append(compute_segment_mean(attribution, spurious_mask))

        trust_concat.append(trust)
        attr_concat.append(attribution)
        label_concat.append(causal_mask.astype(int))

        progress.set_postfix({
            'trust_causal': f"{trust_causal[-1]:.3f}",
            'trust_spu': f"{trust_spurious[-1]:.3f}"
        })

    trust_concat = np.concatenate(trust_concat)
    attr_concat = np.concatenate(attr_concat)
    label_concat = np.concatenate(label_concat)
    precision_trust, recall_trust, _ = precision_recall_curve(label_concat, trust_concat)
    precision_attr, recall_attr, _ = precision_recall_curve(label_concat, attr_concat)

    def summarize(arr):
        return float(np.mean(arr)), float(np.std(arr))

    trust_pref_ratio = float(np.mean(np.array(trust_causal) > np.array(trust_spurious)))
    attr_pref_ratio = float(np.mean(np.array(attr_causal) > np.array(attr_spurious)))

    metrics = {
        'trust_causal_mean': summarize(trust_causal),
        'trust_spurious_mean': summarize(trust_spurious),
        'attr_causal_mean': summarize(attr_causal),
        'attr_spurious_mean': summarize(attr_spurious),
        'trust_prefers_causal_ratio': trust_pref_ratio,
        'attr_prefers_causal_ratio': attr_pref_ratio,
        'roc_trust': roc_auc_score(label_concat, trust_concat),
        'roc_attr': roc_auc_score(label_concat, attr_concat),
        'aupr_trust': average_precision_score(label_concat, trust_concat),
        'aupr_attr': average_precision_score(label_concat, attr_concat),
        'precision_trust': precision_trust,
        'recall_trust': recall_trust,
        'precision_attr': precision_attr,
        'recall_attr': recall_attr
    }
    return metrics


def visualize_batch_summary(metrics: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：因果 vs 伪相关均值 (Attribution & Trust)
    labels = ['Causal', 'Spurious']
    trust_means = [metrics['trust_causal_mean'][0], metrics['trust_spurious_mean'][0]]
    trust_stds = [metrics['trust_causal_mean'][1], metrics['trust_spurious_mean'][1]]
    attr_means = [metrics['attr_causal_mean'][0], metrics['attr_spurious_mean'][0]]
    attr_stds = [metrics['attr_causal_mean'][1], metrics['attr_spurious_mean'][1]]

    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, trust_means, width, yerr=trust_stds, capsize=4, label='Trust', color='tab:green')
    axes[0].bar(x + width / 2, attr_means, width, yerr=attr_stds, capsize=4, label='Attribution', color='tab:blue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Mean Score')
    axes[0].set_title('Causal vs Spurious (Mean ± Std)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 右图：Precision-Recall 曲线
    axes[1].plot(metrics['recall_trust'], metrics['precision_trust'], label=f'Trust (AUPR={metrics["aupr_trust"]:.3f})', linewidth=2)
    axes[1].plot(metrics['recall_attr'], metrics['precision_attr'], label=f'Attribution (AUPR={metrics["aupr_attr"]:.3f})', linewidth=2)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle('Trust vs Attribution - Batch Summary', y=0.95)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset, train_loader, test_loader = build_dataloaders()

    model = SimpleCNN(length=dataset.length, num_classes=2).to(device)
    model = train_model(model, train_loader, device=device, epochs=5)
    model.eval()

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
    batch_metrics = evaluate_over_batch(pipeline, test_loader, dataset, device, max_samples=64)
    print("\nBatch trust/attribution statistics (mean, std):")
    for key, value in batch_metrics.items():
        if isinstance(value, tuple):
            print(f"  {key}: mean={value[0]:.4f}, std={value[1]:.4f}")
        elif hasattr(value, 'shape'):
            if value.ndim == 0:
                print(f"  {key}: {float(value):.4f}")
            elif value.size == 1:
                print(f"  {key}: {float(value.item()):.4f}")
            else:
                print(f"  {key}: {np.array2string(value, precision=4)}")
        else:
            print(f"  {key}: {value:.4f}")

    perturb_stats = run_trust_perturbation_protocol(
        pipeline,
        model,
        test_loader,
        device=device,
        max_samples=128,
        noise_std=0.01,
        topk_ratio=0.1,
        overlap_thresh=0.5
    )
    print("\nTrust perturbation protocol summary:")
    print(f"  mean overlap: {perturb_stats['mean_overlap']:.4f}")
    print(f"  mean trust delta: {perturb_stats['mean_trust_delta']:.4f}")
    print(f"  case counts: {perturb_stats['case_counts']}")

    visualize_batch_summary(batch_metrics, save_path="unified_results/synthetic_batch_summary.png")

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
