"""Motif-based attack injection: taxonomy network_role -> concrete graph shapes.

RedLab_1 injected fraud per-victim, as event sequences with graph-derived
FEATURES computed after the fact (distinct users per device, etc). This
module instead builds actual graph TOPOLOGY as the primary artefact: a
campaign IS a subgraph - a fan-out star, a fan-in star, or a layering chain -
not a set of transactions that happen to share entities.

Every edge in the output carries (src_id, dst_id, timestamp, amount), the
same schema the legitimate world's user->merchant edges use once passed
through `to_graph_schema`, so the combined graph is one consistent structure
regardless of whether an edge is a real purchase or a laundering hop.

Only vectors with network_role != NONE are rendered here - the other ~28
vectors in the taxonomy describe real fraud too, but have no defining
multi-hop network shape, and this solution's differentiation is entirely in
how it handles the ones that do.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .fraud_profile import FraudProfile
from .world import PaymentWorld
from ..taxonomy.loader import Taxonomy
from ..taxonomy.schema import AttackVector, Maturity, NetworkRole, TemporalShape

MATURITY_WEIGHT = {Maturity.OBSERVED: 1.0, Maturity.EMERGING: 0.6, Maturity.PROJECTED: 0.3}


def to_graph_schema(legit: pd.DataFrame) -> pd.DataFrame:
    """Add uniform src/dst columns to the legitimate transaction table."""
    out = legit.copy()
    out["src_id"] = out["user_id"]
    out["dst_id"] = out["merchant_id"]
    out["src_type"] = "user"
    out["dst_type"] = "merchant"
    out["network_role"] = "none"
    out["motif_id"] = None
    out["hop_index"] = -1
    return out


@dataclass
class MotifConfig:
    seed: int = 2024
    target_fraud_edge_rate: float = 0.015   # share of edges that are fraud
    min_campaigns_per_vector: int = 3
    max_campaigns_per_vector: int = 150
    victim_device_share: float = 0.0        # unused here; kept for interface parity


@dataclass
class MotifInjector:
    world: PaymentWorld
    taxonomy: Taxonomy
    fraud_profile: FraudProfile
    cfg: MotifConfig = field(default_factory=MotifConfig)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)
        self._mule_seq = 0

    def _new_mules(self, n: int) -> np.ndarray:
        ids = np.array([f"X{self._mule_seq + i:07d}" for i in range(n)], dtype=object)
        self._mule_seq += n
        return ids

    def _timestamps(self, shape: TemporalShape, start_day: int, duration: int, n: int,
                    start_date: pd.Timestamp) -> pd.DatetimeIndex:
        rng = self.rng
        if shape == TemporalShape.BURST:
            offs = start_day + np.sort(rng.exponential(0.03, n).cumsum())
        elif shape == TemporalShape.SLOW_DRIP:
            offs = start_day + np.sort(rng.uniform(0, duration, n))
        else:
            offs = start_day + np.sort(rng.uniform(0, max(duration, 0.1), n))
        days = np.floor(offs).astype(int)
        hours = rng.integers(0, 24, n)
        return (start_date + pd.to_timedelta(days, unit="D")
                + pd.to_timedelta(hours, unit="h")
                + pd.to_timedelta(rng.integers(0, 60, n), unit="m"))

    def _amounts(self, n: int, anchor: np.ndarray) -> np.ndarray:
        """Fraud amounts anchored to victim baselines, same discipline RedLab_1
        validated: hides beneath the victim's own ceiling in ~99% of cases."""
        rng = self.rng
        fp = self.fraud_profile
        q = rng.uniform(0.25, 0.75, n)
        amt = fp.ratio_at(q) * np.maximum(anchor, 5.0)
        return np.round(np.maximum(amt, 0.5), 2)

    # -- motif renderers -----------------------------------------------------

    def _fan_out(self, v: AttackVector, source: str, victim_baseline: float,
                n_targets: int, start_day: int, duration: int, start_date, mule_side: bool
                ) -> pd.DataFrame:
        targets = self._new_mules(n_targets) if mule_side else self.world.u_id[
            self.rng.choice(len(self.world.u_id), size=min(n_targets, len(self.world.u_id)),
                            replace=False)]
        ts = self._timestamps(v.simulation.temporal_shape, start_day, duration, len(targets), start_date)
        amt = self._amounts(len(targets), np.full(len(targets), victim_baseline))
        return pd.DataFrame({
            "src_id": source, "dst_id": targets,
            "src_type": "attacker", "dst_type": "mule" if mule_side else "user",
            "timestamp": ts, "amount": amt,
        })

    def _fan_in(self, v: AttackVector, sink: str, victim_baselines: np.ndarray,
               start_day: int, duration: int, start_date) -> pd.DataFrame:
        n = len(victim_baselines)
        ts = self._timestamps(v.simulation.temporal_shape, start_day, duration, n, start_date)
        amt = self._amounts(n, victim_baselines)
        sources = self.world.u_id[self.rng.choice(len(self.world.u_id), size=n, replace=False)]
        return pd.DataFrame({
            "src_id": sources, "dst_id": sink,
            "src_type": "user", "dst_type": "mule",
            "timestamp": ts, "amount": amt,
        })

    def _layering_chain(self, v: AttackVector, source_amount: float, n_hops: int,
                        start_day: int, duration: int, start_date) -> pd.DataFrame:
        """source -> hop_1 -> hop_2 -> ... -> hop_n -> collection, with value
        conserved across hops minus a small skim, timed as a fast sequence."""
        rng = self.rng
        nodes = self._new_mules(n_hops + 2)  # originator, hops..., collection
        skim = rng.uniform(0.02, 0.08, n_hops + 1)
        amounts = [source_amount]
        for s in skim:
            amounts.append(amounts[-1] * (1 - s))
        gap_days = duration / max(n_hops + 1, 1)
        rows = []
        for i in range(n_hops + 1):
            ts = self._timestamps(TemporalShape.BURST, start_day + i * gap_days,
                                  max(gap_days, 0.2), 1, start_date)
            rows.append({"src_id": nodes[i], "dst_id": nodes[i + 1],
                        "src_type": "mule" if i > 0 else "attacker", "dst_type": "mule",
                        "timestamp": ts[0], "amount": round(amounts[i], 2)})
        return pd.DataFrame(rows)

    # -- driver ---------------------------------------------------------------

    def inject(self, legit: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
        graph_vectors = [v for v in self.taxonomy if v.network_role != NetworkRole.NONE]
        if not graph_vectors:
            raise RuntimeError("no network_role-tagged vectors in taxonomy")

        rng = self.rng
        base = legit[legit.amount > 0].groupby("user_id").amount.median()
        start_date = pd.Timestamp(self.world.cfg.start_date)
        horizon = self.world.cfg.days
        n_legit_edges = len(legit)
        target_edges = int(n_legit_edges * self.cfg.target_fraud_edge_rate /
                           max(1 - self.cfg.target_fraud_edge_rate, 1e-9))
        per_vector_budget = target_edges / len(graph_vectors)

        frames: List[pd.DataFrame] = []
        campaigns: List[dict] = []
        cid = 0

        for v in graph_vectors:
            w = MATURITY_WEIGHT[v.maturity]
            n_camp = int(np.clip(round(per_vector_budget * w /
                                       max(np.mean(v.simulation.events_per_campaign), 4)),
                                 self.cfg.min_campaigns_per_vector,
                                 self.cfg.max_campaigns_per_vector))
            lo_e, hi_e = v.simulation.events_per_campaign
            lo_d, hi_d = v.simulation.duration_days

            for _ in range(n_camp):
                cid += 1
                n_ev = int(np.clip(rng.integers(lo_e, hi_e + 1), 3, 60))
                dur = int(np.clip(rng.integers(lo_d, hi_d + 1), 1, horizon - 1))
                start = int(rng.integers(0, max(horizon - dur, 1)))
                victim = base.sample(1, random_state=int(rng.integers(1e9))).index[0]
                baseline = float(base.loc[victim])
                attacker = f"A{cid:06d}"

                if v.network_role in (NetworkRole.FAN_OUT, NetworkRole.ORIGINATOR):
                    frame = self._fan_out(v, attacker, baseline, n_ev, start, dur, start_date,
                                          mule_side=(v.network_role == NetworkRole.ORIGINATOR))
                elif v.network_role in (NetworkRole.FAN_IN, NetworkRole.COLLECTION_POINT):
                    victims = base.sample(min(n_ev, len(base)),
                                          random_state=int(rng.integers(1e9)))
                    frame = self._fan_in(v, attacker, victims.to_numpy(), start, dur, start_date)
                elif v.network_role == NetworkRole.LAYERING_HOP:
                    n_hops = int(np.clip(n_ev // 4, 2, 8))
                    frame = self._layering_chain(v, baseline * rng.uniform(3, 12), n_hops,
                                                 start, dur, start_date)
                else:
                    continue

                frame["is_fraud"] = 1
                frame["attack_id"] = v.id
                frame["network_role"] = v.network_role.value
                frame["motif_id"] = f"C{cid:06d}"
                frame["hop_index"] = np.arange(len(frame))
                frames.append(frame)
                campaigns.append({"campaign_id": f"C{cid:06d}", "vector_id": v.id,
                                  "network_role": v.network_role.value,
                                  "n_edges": len(frame), "start_day": start,
                                  "duration_days": dur})

        fraud = pd.concat(frames, ignore_index=True)
        fraud["txn_id"] = [f"G{i:07d}" for i in range(len(fraud))]
        fraud["mcc"] = "4829"
        fraud["channel"] = "Online Transaction"

        combined = pd.concat([to_graph_schema(legit), fraud], ignore_index=True, sort=False)
        combined = combined.sort_values("timestamp", ignore_index=True)
        return combined, campaigns
