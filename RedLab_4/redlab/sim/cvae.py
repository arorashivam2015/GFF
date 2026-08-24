"""A conditional tabular VAE - the trained generative model this solution's
hypothesis is actually about. Conditioned on attack-vector identity (or a
dedicated LEGIT token) so one trained model can emit any of the 42 taxonomy
vectors, or the legitimate population, on demand.

Kept deliberately small: two hidden layers, 64/32 units, a 12-dim latent
space. This is not an architecture-scale claim - it is a direct lesson from
this portfolio's own history (RedLab_2's GNN stalled the build for two days
under a training loop with no early smoke-test). Every piece here is smoke-
tested on a tiny batch before any real training run, and epoch counts are
small and fixed, never an open-ended loop.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 12
HIDDEN = (64, 32)


class ConditionalVAE(nn.Module):
    def __init__(self, n_mcc: int, n_channel: int, n_conditions: int,
                mcc_emb: int = 8, chan_emb: int = 4, cond_emb: int = 8):
        super().__init__()
        self.mcc_embed = nn.Embedding(n_mcc, mcc_emb)
        self.chan_embed = nn.Embedding(n_channel, chan_emb)
        self.cond_embed = nn.Embedding(n_conditions, cond_emb)

        # numeric(3: log_amount, hour_sin, hour_cos) + mcc_emb + chan_emb + cond_emb
        in_dim = 3 + mcc_emb + chan_emb + cond_emb
        h1, h2 = HIDDEN
        self.enc = nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.mu = nn.Linear(h2, LATENT_DIM)
        self.logvar = nn.Linear(h2, LATENT_DIM)

        dec_in = LATENT_DIM + cond_emb
        self.dec = nn.Sequential(nn.Linear(dec_in, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU())
        self.out_numeric = nn.Linear(h1, 3)     # log_amount, hour_sin, hour_cos
        self.out_mcc = nn.Linear(h1, n_mcc)
        self.out_chan = nn.Linear(h1, n_channel)

    def encode(self, numeric, mcc_idx, chan_idx, cond_idx):
        x = torch.cat([numeric, self.mcc_embed(mcc_idx), self.chan_embed(chan_idx),
                      self.cond_embed(cond_idx)], dim=-1)
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, cond_idx):
        x = torch.cat([z, self.cond_embed(cond_idx)], dim=-1)
        h = self.dec(x)
        return self.out_numeric(h), self.out_mcc(h), self.out_chan(h)

    def forward(self, numeric, mcc_idx, chan_idx, cond_idx):
        mu, logvar = self.encode(numeric, mcc_idx, chan_idx, cond_idx)
        z = self.reparameterize(mu, logvar)
        num_out, mcc_logits, chan_logits = self.decode(z, cond_idx)
        return num_out, mcc_logits, chan_logits, mu, logvar


def vae_loss(num_out, mcc_logits, chan_logits, mu, logvar,
            numeric, mcc_idx, chan_idx, kl_weight: float = 0.1):
    recon_numeric = F.mse_loss(num_out, numeric)
    recon_mcc = F.cross_entropy(mcc_logits, mcc_idx)
    recon_chan = F.cross_entropy(chan_logits, chan_idx)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_numeric + recon_mcc + recon_chan + kl_weight * kl
    return total, {"recon_numeric": recon_numeric.item(), "recon_mcc": recon_mcc.item(),
                  "recon_chan": recon_chan.item(), "kl": kl.item()}


@dataclass
class VAEEncoding:
    """Feature encoding shared between training and sampling, so both sides
    of the fidelity comparison use identical preprocessing."""
    mcc_to_idx: Dict[str, int]
    idx_to_mcc: List[str]
    chan_to_idx: Dict[str, int]
    idx_to_chan: List[str]
    cond_to_idx: Dict[str, int]
    idx_to_cond: List[str]
    amount_log_mean: float
    amount_log_std: float

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "VAEEncoding":
        mccs = sorted(df.mcc.unique())
        chans = sorted(df.channel.unique())
        conds = ["LEGIT"] + sorted(df.attack_id.dropna().unique())
        logamt = np.log1p(df.amount.clip(lower=0))
        return cls(
            mcc_to_idx={m: i for i, m in enumerate(mccs)}, idx_to_mcc=mccs,
            chan_to_idx={c: i for i, c in enumerate(chans)}, idx_to_chan=chans,
            cond_to_idx={c: i for i, c in enumerate(conds)}, idx_to_cond=conds,
            amount_log_mean=float(logamt.mean()), amount_log_std=float(logamt.std() + 1e-6),
        )

    def encode_frame(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor,
                                                       torch.Tensor, torch.Tensor]:
        logamt = (np.log1p(df.amount.clip(lower=0)) - self.amount_log_mean) / self.amount_log_std
        hour = df.hour.to_numpy(dtype=float)
        hour_sin, hour_cos = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        numeric = torch.tensor(np.stack([logamt, hour_sin, hour_cos], axis=1), dtype=torch.float32)
        mcc_idx = torch.tensor(df.mcc.map(self.mcc_to_idx).to_numpy(dtype=np.int64))
        chan_idx = torch.tensor(df.channel.map(self.chan_to_idx).to_numpy(dtype=np.int64))
        cond = df.attack_id.fillna("LEGIT")
        cond_idx = torch.tensor(cond.map(self.cond_to_idx).to_numpy(dtype=np.int64))
        return numeric, mcc_idx, chan_idx, cond_idx

    def decode_numeric(self, num_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logamt = num_out[:, 0] * self.amount_log_std + self.amount_log_mean
        amount = np.expm1(np.clip(logamt, 0, 15))
        hour_sin, hour_cos = num_out[:, 1], num_out[:, 2]
        hour = (np.arctan2(hour_sin, hour_cos) / (2 * np.pi)) % 1.0 * 24
        return amount, hour.round().astype(int) % 24


@torch.no_grad()
def sample(model: ConditionalVAE, enc: VAEEncoding, condition: str, n: int,
          seed: int = 0) -> pd.DataFrame:
    model.eval()
    torch.manual_seed(seed)
    cond_idx = torch.full((n,), enc.cond_to_idx[condition], dtype=torch.long)
    z = torch.randn(n, LATENT_DIM)
    num_out, mcc_logits, chan_logits = model.decode(z, cond_idx)
    amount, hour = enc.decode_numeric(num_out.numpy())
    mcc = np.array(enc.idx_to_mcc)[mcc_logits.argmax(-1).numpy()]
    channel = np.array(enc.idx_to_chan)[chan_logits.argmax(-1).numpy()]
    is_fraud = 0 if condition == "LEGIT" else 1
    return pd.DataFrame({"amount": amount, "hour": hour, "mcc": mcc, "channel": channel,
                         "is_fraud": is_fraud, "attack_id": condition})
