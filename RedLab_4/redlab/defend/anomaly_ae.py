"""The primary detector: a small autoencoder trained ONLY on legitimate
transactions, zero fraud labels, reconstruction error as the anomaly score.

DESIGNED TO BE THE LOOP'S DIRECT ADVERSARIAL TARGET. The feature
representation here - numeric(3) concatenated with one-hot mcc and one-hot
channel - is deliberately the SAME continuous space the generator's decoder
already outputs (mcc/channel as softmax probabilities rather than a
discrete embedding lookup). That shared representation is what makes a
literal, differentiable generator -> detector pipeline possible in
redlab/loop/adversarial.py: the generator's own soft outputs feed directly
into this detector's encoder, with no non-differentiable argmax in between.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

AE_HIDDEN = (48, 16)


class Autoencoder(nn.Module):
    def __init__(self, in_dim: int, latent: int = 8):
        super().__init__()
        h1, h2 = AE_HIDDEN
        self.enc = nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(),
                                 nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, latent))
        self.dec = nn.Sequential(nn.Linear(latent, h2), nn.ReLU(),
                                 nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, in_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.enc(x)
        return self.dec(z), z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-row MSE - differentiable w.r.t. x, which is exactly the
        property the adversarial loop needs to backprop through a FROZEN
        copy of this model into an upstream generator's parameters."""
        recon, _ = self(x)
        return ((recon - x) ** 2).mean(dim=-1)


def featurize(df: pd.DataFrame, mcc_to_idx: dict, chan_to_idx: dict,
             amount_log_mean: float, amount_log_std: float) -> torch.Tensor:
    """The same continuous representation the VAE's decoder emits, built
    here from real (discrete) columns via one-hot encoding."""
    logamt = (np.log1p(df.amount.clip(lower=0)) - amount_log_mean) / amount_log_std
    hour = df.hour.to_numpy(dtype=float)
    hour_sin, hour_cos = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    numeric = np.stack([logamt, hour_sin, hour_cos], axis=1)

    n_mcc, n_chan = len(mcc_to_idx), len(chan_to_idx)
    mcc_oh = np.zeros((len(df), n_mcc), dtype=np.float32)
    mcc_oh[np.arange(len(df)), df.mcc.map(mcc_to_idx).to_numpy()] = 1.0
    chan_oh = np.zeros((len(df), n_chan), dtype=np.float32)
    chan_oh[np.arange(len(df)), df.channel.map(chan_to_idx).to_numpy()] = 1.0

    x = np.concatenate([numeric, mcc_oh, chan_oh], axis=1)
    return torch.tensor(x, dtype=torch.float32)


@dataclass
class AnomalyDetector:
    in_dim: int
    epochs: int = 15
    lr: float = 1e-3
    seed: int = 0
    model: Autoencoder = field(default=None, init=False)
    threshold: float = field(default=None, init=False)

    def fit(self, legit_x: torch.Tensor, calib_frac: float = 0.15) -> "AnomalyDetector":
        torch.manual_seed(self.seed)
        n = legit_x.shape[0]
        perm = torch.randperm(n)
        n_calib = int(n * calib_frac)
        calib_x, fit_x = legit_x[perm[:n_calib]], legit_x[perm[n_calib:]]

        self.model = Autoencoder(self.in_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            perm2 = torch.randperm(len(fit_x))
            total = 0.0
            for i in range(0, len(perm2), 512):
                idx = perm2[i:i + 512]
                opt.zero_grad()
                loss = self.model.reconstruction_error(fit_x[idx]).mean()
                loss.backward()
                opt.step()
                total += loss.item()

        # Split-conformal threshold, same discipline as RedLab_6: a held-out
        # legit slice the model never fit on, not a training-set quantile.
        with torch.no_grad():
            calib_scores = self.model.reconstruction_error(calib_x).numpy()
        self.threshold = float(np.quantile(calib_scores, 0.995))
        return self

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        return self.model.reconstruction_error(x).numpy()

    def flag(self, x: torch.Tensor) -> np.ndarray:
        return self.score(x) >= self.threshold
