"""D2: the adversarial curriculum - the closed loop itself.

Red and blue take turns. Each round the attacker searches the knobs a real
fraudster actually controls for settings that evade the CURRENT detector; the
detector then retrains on what got through. Evasion rate per round is the
headline output.

EVASION MUST BE COSTLY
----------------------
An attacker who evades by shrinking every transaction to a rounding error has
not won - they have stopped committing profitable fraud. So fitness is not
evasion rate alone but

    fitness = evasion_rate * value_retention

where value_retention is extracted value relative to the unconstrained attack.
A mutation that halves detection while cutting takings by 90% scores worse than
no mutation at all. Without this term the search degenerates immediately into
micro-amounts, which is a defence win being scored as an attacker win.

WHAT THE ATTACKER CONTROLS
--------------------------
Only observable-side knobs, matching PF-ADV-002's threat model: where in the
amount distribution to sit, how many events per campaign, how much to reuse
entities, whether to operate from the victim's device, and campaign tempo. The
attacker never sees model internals, gradients, or the feature matrix - it only
observes the score its transactions receive, which is what a decline-response
oracle (PF-ADV-001) leaks in practice.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from ..defend.detect import Detector, temporal_split
from ..defend.features import build_features
from ..sim.attacks import AttackConfig, AttackInjector
from ..sim.world import PaymentWorld
from ..taxonomy.loader import Taxonomy


@dataclass(frozen=True)
class AttackGenome:
    """The attacker's controllable parameters."""

    amount_shift: float = 0.0        # shift of the amount quantile band, [-0.4, +0.4]
    amount_width: float = 1.0        # widen/narrow the band, [0.4, 1.6]
    reuse_delta: float = 0.0         # change to entity reuse, [-0.4, +0.4]
    victim_device_share: float = 0.38
    events_scale: float = 1.0        # campaign size multiplier, [0.3, 1.6]
    burst_spread: float = 1.0        # stretch campaign duration, [0.5, 3.0]

    def mutate(self, rng: np.random.Generator, strength: float = 1.0) -> "AttackGenome":
        return AttackGenome(
            amount_shift=float(np.clip(
                self.amount_shift + rng.normal(0, 0.12 * strength), -0.40, 0.40)),
            amount_width=float(np.clip(
                self.amount_width * np.exp(rng.normal(0, 0.18 * strength)), 0.4, 1.6)),
            reuse_delta=float(np.clip(
                self.reuse_delta + rng.normal(0, 0.12 * strength), -0.40, 0.40)),
            victim_device_share=float(np.clip(
                self.victim_device_share + rng.normal(0, 0.12 * strength), 0.0, 0.95)),
            events_scale=float(np.clip(
                self.events_scale * np.exp(rng.normal(0, 0.22 * strength)), 0.3, 1.6)),
            burst_spread=float(np.clip(
                self.burst_spread * np.exp(rng.normal(0, 0.25 * strength)), 0.5, 3.0)),
        )

    def as_dict(self) -> Dict[str, float]:
        return {"amount_shift": self.amount_shift, "amount_width": self.amount_width,
                "reuse_delta": self.reuse_delta,
                "victim_device_share": self.victim_device_share,
                "events_scale": self.events_scale, "burst_spread": self.burst_spread}


class GenomeInjector(AttackInjector):
    """Attack injector whose spec rendering is warped by a genome."""

    genome: AttackGenome = AttackGenome()

    def _amounts(self, profile, n, base):
        from ..sim.attacks import AMOUNT_QUANTILE_BAND

        g = self.genome
        lo, hi = AMOUNT_QUANTILE_BAND.get(profile, (0.25, 0.60))
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * g.amount_width
        lo2 = float(np.clip(mid + g.amount_shift - half, 0.0, 0.999))
        hi2 = float(np.clip(mid + g.amount_shift + half, lo2 + 0.01, 1.0))

        fp = self.fraud_profile
        rng = self.rng
        anchor = base.mean_amt.to_numpy(dtype=float)
        vmax = base.max_amt.to_numpy(dtype=float)
        q = rng.uniform(lo2, hi2, n)
        amt = fp.ratio_at(q) * np.maximum(anchor, 0.5)
        breach = rng.random(n) < fp.share_above_victim_max
        amt = np.where(breach, amt, np.minimum(amt, vmax * rng.uniform(0.55, 1.0, n)))
        return np.round(np.maximum(amt, 0.25), 2)


class RoundResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round: int
    evasion_rate: float
    value_retention: float
    fitness: float
    detector_recall: float
    detector_precision: float
    n_fraud: int
    genome: Dict[str, float]


def _apply_genome_to_specs(tax: Taxonomy, g: AttackGenome):
    """Warp simulation specs by the genome, returning a shallow patched copy."""
    for v in tax:
        sim = v.simulation
        sim.entity_reuse = float(np.clip(sim.entity_reuse + g.reuse_delta, 0.0, 1.0))
        lo, hi = sim.events_per_campaign
        sim.events_per_campaign = [max(int(lo * g.events_scale), 1),
                                   max(int(hi * g.events_scale), 2)]
        dlo, dhi = sim.duration_days
        sim.duration_days = [max(int(dlo * g.burst_spread), 1),
                             max(int(dhi * g.burst_spread), 2)]
    return tax


def _inject(world: PaymentWorld, legit: pd.DataFrame, genome: AttackGenome,
            seed: int) -> pd.DataFrame:
    tax = _apply_genome_to_specs(Taxonomy.load(), genome)
    cfg = AttackConfig(seed=seed, victim_device_share=genome.victim_device_share)
    inj = GenomeInjector(world, tax, cfg)
    inj.genome = genome
    combined, _ = inj.inject(legit)
    return combined


def _score_attack(det: Detector, frame: pd.DataFrame, fpr: float = 0.005
                  ) -> Tuple[float, float, float]:
    """Return (evasion_rate, recall, precision) at a fixed FPR budget."""
    feats = build_features(frame)
    p = det.score(feats)
    y = feats["is_fraud"].to_numpy()
    neg = p[y == 0]
    thresh = np.quantile(neg, 1 - fpr) if len(neg) else np.inf
    flagged = p >= thresh
    tp = int((flagged & (y == 1)).sum())
    recall = tp / max(int(y.sum()), 1)
    precision = tp / max(int(flagged.sum()), 1)
    return 1.0 - recall, recall, precision


def run_curriculum(world: PaymentWorld, legit: pd.DataFrame, rounds: int = 5,
                   candidates: int = 4, seed: int = 7,
                   verbose: bool = True) -> List[RoundResult]:
    rng = np.random.default_rng(seed)
    genome = AttackGenome()

    # Round 0: baseline attack, baseline detector.
    base_frame = _inject(world, legit, genome, seed)
    base_value = float(base_frame.loc[base_frame.is_fraud == 1, "amount"].abs().sum())
    feats = build_features(base_frame)
    tr, te = temporal_split(feats)
    det = Detector(n_estimators=250).fit(tr)

    results: List[RoundResult] = []
    for r in range(rounds):
        pool = [genome] + [genome.mutate(rng, strength=1.0 - 0.12 * r)
                           for _ in range(candidates)]
        scored = []
        for cand in pool:
            frame = _inject(world, legit, cand, seed + 100 * r)
            ev, rec, prec = _score_attack(det, frame)
            value = float(frame.loc[frame.is_fraud == 1, "amount"].abs().sum())
            retention = value / max(base_value, 1.0)
            fitness = ev * min(retention, 1.5)
            scored.append((fitness, ev, retention, rec, prec, cand, frame))

        scored.sort(key=lambda t: -t[0])
        fitness, ev, retention, rec, prec, genome, best_frame = scored[0]

        results.append(RoundResult(
            round=r, evasion_rate=ev, value_retention=retention, fitness=fitness,
            detector_recall=rec, detector_precision=prec,
            n_fraud=int(best_frame.is_fraud.sum()), genome=genome.as_dict()))
        if verbose:
            print(f"  round {r}: evasion {ev*100:5.1f}%  value {retention*100:5.1f}%  "
                  f"fitness {fitness:.3f}  recall {rec*100:5.1f}%")

        # Blue team retrains on what got through.
        feats = build_features(best_frame)
        tr, te = temporal_split(feats)
        det = Detector(n_estimators=250).fit(tr)

    return results
