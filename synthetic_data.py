import torch
from torch.utils.data import Dataset
import numpy as np


class CausalSpuriousTimeSeriesDataset(Dataset):
    """
    Synthetic Time Series Dataset with:
    - causal segments
    - spurious segments
    - noise segments

    Designed for:
    - attribution uncertainty
    - cross-view consistency
    - trust validation
    """

    def __init__(
        self,
        n_samples: int = 1000,
        length: int = 200,
        causal_range=(40, 70),
        spurious_range=(120, 150),
        noise_std: float = 0.5,
        causal_strength: float = 2.0,
        spurious_strength: float = 1.5,
        spurious_flip_prob: float = 0.3,
        seed: int = 42
    ):
        """
        Args:
            n_samples: number of samples
            length: time series length
            causal_range: (start, end) of causal segment
            spurious_range: (start, end) of spurious segment
            noise_std: background noise std
            causal_strength: signal strength of causal segment
            spurious_strength: signal strength of spurious segment
            spurious_flip_prob: probability to flip spurious-label correlation
        """
        super().__init__()
        self.n_samples = n_samples
        self.length = length
        self.causal_range = causal_range
        self.spurious_range = spurious_range
        self.noise_std = noise_std
        self.causal_strength = causal_strength
        self.spurious_strength = spurious_strength
        self.spurious_flip_prob = spurious_flip_prob

        rng = np.random.RandomState(seed)

        self.data = []
        self.labels = []

        for _ in range(n_samples):
            x = rng.randn(length) * noise_std

            # binary label
            y = rng.randint(0, 2)

            # ------------------
            # causal segment
            # ------------------
            causal_signal = causal_strength if y == 1 else -causal_strength
            x[causal_range[0]: causal_range[1]] += causal_signal

            # ------------------
            # spurious segment
            # ------------------
            spurious_y = y
            if rng.rand() < spurious_flip_prob:
                spurious_y = 1 - y  # flip correlation

            spurious_signal = spurious_strength if spurious_y == 1 else -spurious_strength
            x[spurious_range[0]: spurious_range[1]] += spurious_signal

            self.data.append(x.astype(np.float32))
            self.labels.append(y)

        self.data = np.stack(self.data)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
            x: [1, length]
            y: scalar
        """
        x = torch.from_numpy(self.data[idx]).unsqueeze(0)
        y = torch.tensor(self.labels[idx])
        return x, y

    # ====== ground-truth masks (for evaluation only) ======

    def get_causal_mask(self):
        mask = np.zeros(self.length, dtype=np.float32)
        mask[self.causal_range[0]: self.causal_range[1]] = 1.0
        return mask

    def get_spurious_mask(self):
        mask = np.zeros(self.length, dtype=np.float32)
        mask[self.spurious_range[0]: self.spurious_range[1]] = 1.0
        return mask
