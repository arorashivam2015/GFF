"""The white-box adversarial loop: the generator's loss function incorporates
the FROZEN detector's output directly, via a real gradient path, not a
black-box parameter search.

This is only possible because of a design decision made back in Generate and
Defend: both the VAE's decoder and the autoencoder detector operate on the
IDENTICAL continuous representation - numeric(3) plus mcc/channel as
probability vectors, never a discrete embedding lookup. That shared space is
what lets the generator's own softmax outputs feed directly into the frozen
detector's encoder with no non-differentiable argmax breaking the gradient.

THREAT MODEL, STATED EXPLICITLY: this is WHITE-BOX. The attacker has direct
gradient access to a differentiable generator AND to a frozen copy of the
deployed detector's weights. That is a strictly stronger assumption than the
BLACK-BOX, query-only attacker RedLab_1 and RedLab_7 model (decline-response
oracle only, no internals). Neither is more "realistic" than the other -
they are different, both legitimate, threat models, and a real portfolio
benefits from having both. Do not read this loop's evasion numbers as
comparable to RedLab_1's black-box hill-climb results; they answer a
different question (what a gradient-empowered attacker can do) against a
different defender (an unsupervised autoencoder, not a supervised GBM).

VALUE RETENTION: fine-tuning is regularised to stay close to the generator's
own un-adversarial output (an L2 penalty in the same continuous space the
detector scores), so evasion cannot degenerate into an unconstrained point
that no longer resembles the conditioned attack vector at all - the same
principle RedLab_1 encoded as evasion_rate x value_retention, adapted here
to a continuous, gradient-based setting rather than a discrete genome.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ..defend.anomaly_ae import Autoencoder
from ..sim.cvae import ConditionalVAE, VAEEncoding, LATENT_DIM


@dataclass
class EvasionResult:
    condition: str
    n_samples: int
    evasion_rate_before: float
    evasion_rate_after: float
    mean_drift: float          # how far fine-tuned outputs moved from the original decode
    rounds: List[Dict[str, float]] = field(default_factory=list)


def _soft_representation(num_out: torch.Tensor, mcc_logits: torch.Tensor,
                         chan_logits: torch.Tensor) -> torch.Tensor:
    """The generator's outputs, converted to the exact continuous space the
    autoencoder detector was trained on - softmax rather than argmax, so the
    whole path stays differentiable."""
    return torch.cat([num_out, F.softmax(mcc_logits, dim=-1),
                      F.softmax(chan_logits, dim=-1)], dim=-1)


def run_evasion(generator: ConditionalVAE, detector: Autoencoder, enc: VAEEncoding,
                condition: str, threshold: float, n_samples: int = 500,
                steps: int = 200, lr: float = 0.02, drift_weight: float = 0.5,
                seed: int = 0) -> EvasionResult:
    torch.manual_seed(seed)
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)   # frozen: no gradient updates to the detector, ever

    cond_idx = torch.full((n_samples,), enc.cond_to_idx[condition], dtype=torch.long)
    z0 = torch.randn(n_samples, LATENT_DIM)

    with torch.no_grad():
        num0, mcc0, chan0 = generator.decode(z0, cond_idx)
        x0 = _soft_representation(num0, mcc0, chan0)
        err0 = detector.reconstruction_error(x0)
        before = float((err0 >= threshold).float().mean())

    # A free latent vector, optimised directly - not the generator's own
    # weights. This is equivalent in spirit to fine-tuning the generator's
    # LAST layer for this one condition, but far cheaper and just as much a
    # genuine gradient-based attack: the optimisation target is still "make
    # the frozen detector's reconstruction error small," backpropagated
    # through the (frozen) generator decoder and the (frozen) detector alike.
    z = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    rounds = []
    for step in range(steps):
        opt.zero_grad()
        num_out, mcc_logits, chan_logits = generator.decode(z, cond_idx)
        x = _soft_representation(num_out, mcc_logits, chan_logits)
        evasion_loss = detector.reconstruction_error(x).mean()
        drift = ((x - x0.detach()) ** 2).mean()
        loss = evasion_loss + drift_weight * drift
        loss.backward()
        opt.step()
        if step % 40 == 0 or step == steps - 1:
            with torch.no_grad():
                err = detector.reconstruction_error(x)
                rate = float((err >= threshold).float().mean())
            rounds.append({"step": step, "evasion_loss": float(evasion_loss.item()),
                          "drift": float(drift.item()), "flagged_rate": rate})

    with torch.no_grad():
        num_f, mcc_f, chan_f = generator.decode(z, cond_idx)
        xf = _soft_representation(num_f, mcc_f, chan_f)
        errf = detector.reconstruction_error(xf)
        after = float((errf >= threshold).float().mean())
        mean_drift = float(((xf - x0) ** 2).mean().sqrt())

    return EvasionResult(
        condition=condition, n_samples=n_samples,
        evasion_rate_before=1 - before, evasion_rate_after=1 - after,
        mean_drift=mean_drift, rounds=rounds)
