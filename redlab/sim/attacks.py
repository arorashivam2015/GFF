"""Attack injection: taxonomy specs -> concrete fraud campaigns in the world.

Fraud is injected INTO the legitimate population rather than generated beside
it, so every attack inherits a real background - real merchants, real devices,
real per-victim spending baselines it has to hide inside. Standalone fraud
generation is what makes synthetic fraud trivially detectable: if the attacker's
amounts are drawn from a different distribution than the victim's own history,
a detector learns the distribution gap rather than the attack.

The engine is generic. It reads `AttackVector.simulation` from the taxonomy and
renders it, so all 42 vectors work without per-vector code. That is the payoff
for making the taxonomy machine-readable: adding a vector to a YAML file is
sufficient to make it simulatable.

RAIL SIMPLIFICATION (stated openly)
-----------------------------------
The calibrated world is card-rail, because the reference corpus is card data.
Vectors on UPI, AePS, IMPS and NACH rails are injected using transfer-like
semantics - money-transfer category, P2P beneficiary structure - rather than a
separately calibrated UPI world. The observable signal structure the detector
consumes (amount relative to baseline, velocity, graph topology, timing) does
transfer; the rail label does not. Where a claim depends on UPI-specific
calibration, it is marked as resting on the published-aggregate priors in
calibration.upi_prior(), not on transaction-level data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..taxonomy.loader import Taxonomy
from ..taxonomy.schema import AmountProfile, AttackVector, Maturity, TemporalShape
from .fraud_profile import FraudProfile
from .world import PaymentWorld

# Categories that stand in for transfer-like rails (UPI/IMPS/NEFT/AePS).
TRANSFER_MCC = "4829"
CASH_EQUIVALENT_MCC = {"4829", "6011", "7995", "6051"}

# Each amount profile selects a REGION of the empirically measured
# fraud-amount-to-victim-baseline ratio distribution, rather than applying an
# invented formula. This keeps the aggregate fraud amount distribution matched
# to the reference while still letting profiles differ from one another.
AMOUNT_QUANTILE_BAND = {
    AmountProfile.MICRO_PROBE:      (0.00, 0.12),
    AmountProfile.TYPICAL:          (0.25, 0.60),
    AmountProfile.JUST_UNDER_LIMIT: (0.50, 0.78),
    AmountProfile.ESCALATING:       (0.15, 0.92),   # traversed as a ramp
    AmountProfile.DRAIN:            (0.85, 0.995),
    AmountProfile.ROUND_SUM:        (0.60, 0.95),
}

# Relative volume by maturity: observed attacks are simply more common today
# than projected ones. Without this the corpus over-weights speculative vectors.
MATURITY_WEIGHT = {
    Maturity.OBSERVED: 1.0,
    Maturity.EMERGING: 0.55,
    Maturity.PROJECTED: 0.20,
}


@dataclass
class AttackConfig:
    target_fraud_rate: float = 0.0122   # 10x the reference base rate, see note
    seed: int = 1234
    min_campaigns_per_vector: int = 3
    max_campaigns_per_vector: int = 400
    min_events_per_vector: int = 8
    victim_device_share: float = 0.38
    # Share of fraud events executed from the VICTIM's own device rather than
    # attacker infrastructure. Session hijack, malware and on-device social
    # engineering all operate from the genuine device. Routing 100% of fraud
    # through fresh attacker devices made "unseen device" a perfect oracle.

    # The reference corpus sits at 0.122% fraud. We inject at 1.22% so that
    # every one of 42 vectors - including low-weight projected ones - lands
    # enough events to be learnable and, more importantly, so the
    # leave-one-family-out holdout has enough positives to score. The detector
    # evaluation reweights to the reference prevalence when reporting
    # operating-point metrics, so this does not flatter the results.


@dataclass
class Campaign:
    """One realised instance of an attack vector."""

    campaign_id: str
    vector_id: str
    family: str
    victim_users: List[str]
    attacker_entities: List[str]
    n_events: int
    start_day: int
    duration_days: int


@dataclass
class AttackInjector:
    world: PaymentWorld
    taxonomy: Taxonomy
    cfg: AttackConfig = field(default_factory=AttackConfig)
    fraud_profile: Optional[FraudProfile] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)
        if self.fraud_profile is None:
            self.fraud_profile = FraudProfile.load()

    # -- victim baselines --------------------------------------------------

    def _baselines(self, legit: pd.DataFrame) -> pd.DataFrame:
        """Per-user spending profile. Attacks are scaled against this so that
        fraud amounts are relative to the victim, not absolute."""
        pos = legit[legit.amount > 0]
        b = pos.groupby("user_id").agg(
            median_amt=("amount", "median"),
            mean_amt=("amount", "mean"),
            p90_amt=("amount", lambda s: s.quantile(0.90)),
            max_amt=("amount", "max"),
            n_txn=("amount", "size"),
        )
        return b[b.n_txn >= 10]

    # -- spec renderers ----------------------------------------------------

    def _amounts(self, profile: AmountProfile, n: int, base: pd.DataFrame) -> np.ndarray:
        """Draw amounts by sampling the measured fraud/baseline ratio.

        The ratio distribution comes from 28,619 labelled reference frauds. Each
        profile occupies a band of it, so profiles stay distinguishable from one
        another while the aggregate matches the reference.

        Amounts are then capped at the victim's own historical maximum, except
        for the measured 0.74% of frauds that genuinely exceed it. An earlier
        version drew DRAIN as a multiple of the victim's max, putting 40% of
        fraud above that ceiling against a real rate of under one percent - and
        the resulting corpus was separable at ROC-AUC 0.989 on amount alone.
        """
        rng = self.rng
        fp = self.fraud_profile
        anchor = base.mean_amt.to_numpy(dtype=float)
        vmax = base.max_amt.to_numpy(dtype=float)

        lo, hi = AMOUNT_QUANTILE_BAND.get(profile, (0.25, 0.60))
        if profile == AmountProfile.ESCALATING:
            q = np.linspace(lo, hi, n) * rng.uniform(0.92, 1.08, n)
        else:
            q = rng.uniform(lo, hi, n)

        amt = fp.ratio_at(q) * np.maximum(anchor, 0.5)

        if profile == AmountProfile.ROUND_SUM:
            amt = np.maximum(np.round(amt / 50.0) * 50.0, 50.0)

        # Respect the victim's own ceiling, at the measured breach rate.
        breach = rng.random(n) < fp.share_above_victim_max
        amt = np.where(breach, amt, np.minimum(amt, vmax * rng.uniform(0.55, 1.0, n)))
        return np.round(np.maximum(amt, 0.25), 2)

    def _timestamps(self, shape: TemporalShape, start_day: int, duration: int,
                    n: int, start_date: pd.Timestamp) -> pd.DatetimeIndex:
        rng = self.rng

        if shape == TemporalShape.BURST:
            t0 = start_day + rng.uniform(0, max(duration - 0.5, 0.5))
            offs = t0 + np.sort(rng.exponential(0.02, n).cumsum())
        elif shape == TemporalShape.SLOW_DRIP:
            offs = start_day + np.sort(rng.uniform(0, duration, n))
        elif shape == TemporalShape.DORMANT_THEN_SPIKE:
            # Long quiet build-up, then the bust-out.
            k = max(int(n * 0.75), 1)
            quiet = start_day + np.sort(rng.uniform(0, duration * 0.85, k))
            spike = start_day + duration * 0.9 + np.sort(
                rng.exponential(0.05, n - k).cumsum()) if n > k else np.array([])
            offs = np.concatenate([quiet, spike])
        elif shape == TemporalShape.SUSTAINED_CAMPAIGN:
            offs = start_day + np.sort(rng.uniform(0, duration, n))
        else:  # BUSINESS_HOURS / OFF_HOURS
            offs = start_day + np.sort(rng.uniform(0, duration, n))

        days = np.floor(offs).astype(int)
        frac = offs - days
        if shape == TemporalShape.BUSINESS_HOURS:
            hours = rng.integers(9, 18, n)
        elif shape == TemporalShape.OFF_HOURS:
            hours = rng.choice(np.r_[0:6, 22:24], n)
        else:
            hours = np.clip((frac * 24).astype(int), 0, 23)

        return (start_date + pd.to_timedelta(days, unit="D")
                + pd.to_timedelta(hours, unit="h")
                + pd.to_timedelta(rng.integers(0, 60, n), unit="m"))

    def _n_campaigns(self, v: AttackVector, total_events: int) -> int:
        """Allocate campaign volume by maturity and scalability."""
        w = MATURITY_WEIGHT[v.maturity] * (0.4 + 0.12 * v.scores.scalability)
        mid = float(np.mean(v.simulation.events_per_campaign))
        n = int(round(total_events * w / max(mid, 1.0)))
        return int(np.clip(n, self.cfg.min_campaigns_per_vector,
                           self.cfg.max_campaigns_per_vector))

    # -- injection ---------------------------------------------------------

    def inject(self, legit: pd.DataFrame) -> Tuple[pd.DataFrame, List[Campaign]]:
        rng = self.rng
        base = self._baselines(legit)
        if base.empty:
            raise RuntimeError("no users with enough history to attack")

        users = base.index.to_numpy()
        merchants = legit.merchant_id.unique()
        mcc_of = legit.drop_duplicates("merchant_id").set_index("merchant_id")["mcc"]

        # Fraud category/channel tables, restricted to categories this world
        # actually contains and renormalised.
        fp = self.fraud_profile
        world_mccs = set(legit.mcc.unique())
        self._merch_by_mcc = {m: g.merchant_id.unique()
                              for m, g in legit.groupby("mcc")}

        mix = {k: v for k, v in fp.mcc_mix.items() if k in world_mccs}
        if not mix:
            mix = {k: 1.0 for k in world_mccs}
        self._fraud_mcc = np.array(list(mix))
        w = np.array(list(mix.values()), dtype=float)
        self._fraud_mcc_w = w / w.sum()

        # Transfer-like rails: cash-equivalent categories weighted up, with the
        # rest of the fraud mix retained so the tail does not collapse.
        tw = np.array([4.0 if k in CASH_EQUIVALENT_MCC else 1.0
                       for k in self._fraud_mcc]) * self._fraud_mcc_w
        self._transfer_mcc = self._fraud_mcc
        self._transfer_w = tw / tw.sum()

        # Victim device lookup, for fraud executed on the genuine device.
        self._user_device = (legit.groupby("user_id")["device_id"]
                             .agg(lambda s: s.value_counts().index[0]).to_dict())

        self._fraud_ch = np.array(list(fp.channel_mix))
        cw = np.array(list(fp.channel_mix.values()), dtype=float)
        self._fraud_ch_w = cw / cw.sum()
        start_date = pd.Timestamp(self.world.cfg.start_date)
        horizon = self.world.cfg.days

        # Total fraud events implied by the target rate.
        n_legit = len(legit)
        total_events = int(n_legit * self.cfg.target_fraud_rate /
                           max(1 - self.cfg.target_fraud_rate, 1e-9))

        # Normalise the per-vector allocation to hit the target in aggregate.
        raw = {v.id: self._n_campaigns(v, total_events) *
               float(np.mean(v.simulation.events_per_campaign))
               for v in self.taxonomy}
        scale = total_events / max(sum(raw.values()), 1.0)

        rows: List[pd.DataFrame] = []
        campaigns: List[Campaign] = []
        cid = 0

        for v in self.taxonomy:
            sim = v.simulation
            n_camp = max(1, int(round(self._n_campaigns(v, total_events) * scale)))
            lo_e, hi_e = sim.events_per_campaign
            lo_d, hi_d = sim.duration_days

            for _ in range(n_camp):
                cid += 1
                n_ev = int(rng.integers(lo_e, hi_e + 1))
                n_ev = int(min(n_ev, 600))          # cap so one spec cannot dominate

                # Duration is clipped to fit inside the world's time horizon.
                # Several vectors (e.g. PF-SID-003, up to 400 days) exceed a
                # 240-day world on their own. Unclipped, horizon - dur went
                # negative, start collapsed to 0, and the campaign's tail ran
                # past the world's last generated day - 304 fraud events (1.9%
                # of the corpus) landed in a period with no legitimate traffic
                # at all, which no real payment system produces.
                dur = int(rng.integers(lo_d, hi_d + 1))
                dur = min(dur, max(horizon - 1, 1))
                start = int(rng.integers(0, max(horizon - dur, 1)))

                # Victim count is set by how many events each victim should
                # receive, not by spreading events across as many victims as
                # possible. Reference fraud clusters hard: median 19 events per
                # victim, only 2.2% of victims see a single event. An earlier
                # version produced median 3 and 20.7% singletons, which left
                # velocity features contributing ~0.001 PR-AUC because no
                # victim ever accumulated a burst.
                per_victim = 4.0 + 24.0 * sim.entity_reuse
                n_vic = int(np.clip(round(n_ev / per_victim), 1, 60))
                vics = rng.choice(users, size=min(n_vic, len(users)), replace=False)

                # Attacker-side entities (mules, attacker merchants, devices).
                n_att = int(np.clip(round(n_ev * (1.0 - sim.entity_reuse) * 0.5), 1, 25))
                att_merch = rng.choice(merchants, size=min(n_att, len(merchants)),
                                       replace=False)
                att_dev = np.array([f"AD{cid:06d}_{j}" for j in range(max(n_att // 2, 1))])

                # Assign victims in contiguous runs over the campaign's sorted
                # timeline, so each victim's fraud arrives as a temporally
                # adjacent burst rather than scattered across the whole window.
                # Real compromise looks like a run of hits on one credential.
                ts = self._timestamps(sim.temporal_shape, start, dur, n_ev, start_date)
                order = np.argsort(ts.values)
                run_owner = np.repeat(rng.permutation(len(vics)),
                                      int(np.ceil(n_ev / len(vics))))[:n_ev]
                ev_user = np.empty(n_ev, dtype=object)
                ev_user[order] = vics[run_owner]

                ev_base = base.loc[ev_user]
                amounts = self._amounts(sim.amount_profile, n_ev, ev_base)

                # Category and channel are drawn from the measured fraud
                # distributions, not asserted. Reference fraud spans 98 MCCs
                # with the top one at 16.9% and is 61% online; routing every
                # transfer-rail vector into a single category at 100% online
                # made category and channel separable at ~0.89 AUC on their own.
                is_transfer = any(r.value.startswith(("upi", "imps", "neft", "aeps", "nach"))
                                  for r in v.rails)
                if is_transfer:
                    # Transfer rails skew to cash-equivalent categories, but
                    # still spread rather than collapsing onto one.
                    ev_mcc = rng.choice(self._transfer_mcc, size=n_ev,
                                        p=self._transfer_w)
                else:
                    ev_mcc = rng.choice(self._fraud_mcc, size=n_ev, p=self._fraud_mcc_w)

                ev_merch = np.array([
                    rng.choice(self._merch_by_mcc[m]) if m in self._merch_by_mcc
                    else rng.choice(att_merch) for m in ev_mcc])
                channel = rng.choice(self._fraud_ch, size=n_ev, p=self._fraud_ch_w)

                rows.append(pd.DataFrame({
                    "txn_id": [f"F{cid:06d}_{i:04d}" for i in range(n_ev)],
                    "timestamp": ts,
                    "hour": ts.hour,
                    "dow": ts.dayofweek,
                    "user_id": ev_user,
                    "device_id": np.where(
                        rng.random(n_ev) < self.cfg.victim_device_share,
                        np.array([self._user_device.get(u, att_dev[0])
                                  for u in ev_user], dtype=object),
                        rng.choice(att_dev, size=n_ev)),
                    "merchant_id": ev_merch,
                    "mcc": ev_mcc,
                    "channel": channel,
                    "amount": amounts,
                    "state": np.where(channel == "Online Transaction", "ONLINE", "XX"),
                    "error": "(none)",
                    "is_fraud": 1,
                    "attack_id": v.id,
                }))
                campaigns.append(Campaign(
                    campaign_id=f"C{cid:06d}", vector_id=v.id, family=v.family.value,
                    victim_users=list(map(str, vics)),
                    attacker_entities=list(map(str, att_merch)),
                    n_events=n_ev, start_day=start, duration_days=dur))

        fraud = pd.concat(rows, ignore_index=True)

        # Honour the target rate. The per-vector campaign floor guarantees all
        # 42 specs are represented, but in a small world that floor alone can
        # overshoot the target several-fold. Downsample PER VECTOR rather than
        # globally, so thinning the corpus never silently drops a spec - which
        # would quietly break the leave-one-family-out holdout.
        if len(fraud) > total_events:
            keep = total_events / len(fraud)
            floor = self.cfg.min_events_per_vector
            fraud = (fraud.groupby("attack_id", group_keys=False)
                     .apply(lambda g: g.sample(
                         min(len(g), max(int(round(len(g) * keep)), floor)),
                         random_state=self.cfg.seed))
                     .reset_index(drop=True))

        fraud["state"] = np.where(
            fraud.state.eq("XX"),
            self.world.m_state[rng.integers(0, len(self.world.m_state), len(fraud))],
            fraud.state)

        combined = (pd.concat([legit, fraud], ignore_index=True)
                    .sort_values("timestamp", ignore_index=True))
        return combined, campaigns


def inject_attacks(world: PaymentWorld, legit: pd.DataFrame,
                   taxonomy: Optional[Taxonomy] = None,
                   cfg: Optional[AttackConfig] = None):
    tax = taxonomy or Taxonomy.load()
    return AttackInjector(world, tax, cfg or AttackConfig()).inject(legit)
