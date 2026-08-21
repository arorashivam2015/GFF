"""Fidelity harness: how closely does generated data resemble the target?

Built before the attack generators on purpose. If fidelity scoring arrives
last, you discover on the final day that the whole corpus is trivially
separable from the reference and there is no time left to fix it.

Three families of evidence, deliberately kept distinct because they support
different strengths of claim (see calibration.py for the honesty note):

  MARGINAL     KS distance, Jensen-Shannon divergence on each distribution
               the profile pins down. Necessary, weak on its own - matching
               marginals while destroying joint structure is easy.

  STYLISED     Benford MAD, Zipf alpha, CNP fraud lift. These are properties
               of *real* payment systems from the literature, so conformance
               means something regardless of the reference corpus.

  ADVERSARIAL  Discriminator AUC. Train a classifier to tell generated rows
               from reference rows. AUC 0.5 = indistinguishable, 1.0 = trivially
               separable. This is the metric that catches joint-structure
               failures the marginals miss, and it is the headline number.
"""

from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .calibration import CalibrationProfile, benford_mad, fit_zipf_alpha


class Verdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class FidelityMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    family: str
    value: float
    target: Optional[float] = None
    verdict: Verdict
    interpretation: str

    def line(self) -> str:
        mark = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL"}[self.verdict.value]
        tgt = f"  target {self.target:.4f}" if self.target is not None else ""
        return f"  [{mark}] {self.name:34s} {self.value:9.4f}{tgt}   {self.interpretation}"


class FidelityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: List[FidelityMetric]
    discriminator_auc: Optional[float] = None
    n_generated: int = 0
    n_reference: int = 0
    reference_name: str = ""

    @property
    def failures(self) -> List[FidelityMetric]:
        return [m for m in self.metrics if m.verdict == Verdict.FAIL]

    @property
    def headline(self) -> str:
        if self.discriminator_auc is None:
            return "discriminator not run"
        a = self.discriminator_auc
        if a < 0.60:
            q = "strong - near-indistinguishable from reference"
        elif a < 0.75:
            q = "good - separable but structurally close"
        elif a < 0.90:
            q = "weak - clear systematic differences"
        else:
            q = "poor - trivially separable"
        return f"discriminator AUC {a:.3f} ({q})"

    def render(self) -> str:
        out = ["", "=" * 78, f"FIDELITY REPORT  vs {self.reference_name}",
               f"generated {self.n_generated:,} rows | reference {self.n_reference:,} rows",
               "=" * 78]
        for fam in ("marginal", "stylised", "adversarial"):
            fam_metrics = [m for m in self.metrics if m.family == fam]
            if not fam_metrics:
                continue
            out.append(f"\n{fam.upper()}")
            out.extend(m.line() for m in fam_metrics)
        out.append(f"\nHEADLINE: {self.headline}")
        if self.failures:
            out.append(f"FAILURES: {', '.join(m.name for m in self.failures)}")
        out.append("=" * 78)
        return "\n".join(out)


# --------------------------------------------------------------------------
# Distance helpers
# --------------------------------------------------------------------------


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence, base 2, so the range is [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def aligned_hist(gen: Dict[str, float], ref: Dict[str, float]):
    """Align two categorical distributions onto a shared key set."""
    keys = sorted(set(gen) | set(ref))
    return ([gen.get(k, 0.0) for k in keys], [ref.get(k, 0.0) for k in keys])


def first_digit_counts(values: np.ndarray) -> np.ndarray:
    """Leading-digit histogram, computed numerically.

    Scaling by the power of ten below each value is exact for the magnitudes
    payment amounts occupy, and avoids the string-formatting round trip that
    silently mishandles scientific notation.
    """
    v = np.abs(np.asarray(values, dtype=float))
    v = v[v > 0]
    if v.size == 0:
        return np.zeros(9, dtype=float)
    lead = (v / np.power(10.0, np.floor(np.log10(v)))).astype(int)
    lead = np.clip(lead, 1, 9)
    return np.bincount(lead, minlength=10)[1:10].astype(float)


def _verdict(value: float, warn: float, fail: float) -> Verdict:
    if value >= fail:
        return Verdict.FAIL
    if value >= warn:
        return Verdict.WARN
    return Verdict.PASS


# --------------------------------------------------------------------------
# Profile comparison
# --------------------------------------------------------------------------


def score_against_profile(
    gen: pd.DataFrame,
    profile: CalibrationProfile,
    amount_col: str = "amount",
    mcc_col: str = "mcc",
    hour_col: str = "hour",
    channel_col: str = "channel",
    is_fraud_col: str = "is_fraud",
) -> List[FidelityMetric]:
    """Score a generated frame against the calibration targets."""
    m: List[FidelityMetric] = []
    amt = gen[amount_col].to_numpy(dtype=float)
    pos = amt[amt > 0]

    # --- marginal -------------------------------------------------------
    ref_pcts = np.array([profile.amount.percentiles[k]
                         for k in sorted(profile.amount.percentiles,
                                         key=lambda s: float(s[1:]))])
    ref_q = np.array(sorted(float(k[1:]) for k in profile.amount.percentiles))
    gen_pcts = np.percentile(pos, ref_q)
    # Compare on log scale: amounts span orders of magnitude.
    rel = np.abs(np.log1p(gen_pcts) - np.log1p(ref_pcts)) / np.log1p(ref_pcts)
    m.append(FidelityMetric(
        name="amount_percentile_log_rmse", family="marginal",
        value=float(np.sqrt((rel ** 2).mean())), target=0.0,
        verdict=_verdict(float(np.sqrt((rel ** 2).mean())), 0.10, 0.25),
        interpretation="relative error across the amount quantile curve"))

    if hour_col in gen:
        gh = np.bincount(gen[hour_col].to_numpy(dtype=int), minlength=24)[:24]
        jsd = js_divergence(gh, profile.temporal.hour_of_day)
        m.append(FidelityMetric(
            name="hour_of_day_jsd", family="marginal", value=jsd, target=0.0,
            verdict=_verdict(jsd, 0.02, 0.08),
            interpretation="circadian rhythm divergence"))

    if mcc_col in gen:
        gm = gen[mcc_col].astype(str).value_counts(normalize=True).to_dict()
        p, q = aligned_hist(gm, profile.merchant.mcc_mix)
        jsd = js_divergence(p, q)
        m.append(FidelityMetric(
            name="mcc_mix_jsd", family="marginal", value=jsd, target=0.0,
            verdict=_verdict(jsd, 0.05, 0.15),
            interpretation="merchant-category mix divergence"))

    if channel_col in gen:
        gc = gen[channel_col].astype(str).value_counts(normalize=True).to_dict()
        p, q = aligned_hist(gc, profile.channel_mix)
        jsd = js_divergence(p, q)
        m.append(FidelityMetric(
            name="channel_mix_jsd", family="marginal", value=jsd, target=0.0,
            verdict=_verdict(jsd, 0.03, 0.10),
            interpretation="card-present vs CNP channel mix"))

    # --- stylised facts -------------------------------------------------
    fd = first_digit_counts(pos)
    if fd.sum() > 0:
        mad = benford_mad(fd)
        ref_mad = benford_mad(np.array(profile.amount.benford_first_digit))
        m.append(FidelityMetric(
            name="benford_mad_pp", family="stylised", value=mad, target=ref_mad,
            verdict=_verdict(abs(mad - ref_mad), 0.6, 1.5),
            interpretation=f"Nigrini MAD; reference corpus sits at {ref_mad:.2f}pp"))

    if "merchant_id" in gen:
        counts = gen["merchant_id"].value_counts().to_numpy(dtype=float)
        alpha = fit_zipf_alpha(counts)
        d = abs(alpha - profile.merchant.zipf_alpha)
        m.append(FidelityMetric(
            name="merchant_zipf_alpha", family="stylised", value=alpha,
            target=profile.merchant.zipf_alpha, verdict=_verdict(d, 0.25, 0.60),
            interpretation="merchant popularity concentration"))

    if is_fraud_col in gen and channel_col in gen:
        g = gen.groupby(channel_col)[is_fraud_col].mean()
        online = g[[i for i in g.index if "online" in str(i).lower()]]
        present = g[[i for i in g.index
                     if "swipe" in str(i).lower() or "chip" in str(i).lower()]]
        if len(online) and len(present) and present.mean() > 0:
            lift = float(online.mean() / present.mean())
            ref_lift = profile.cnp_fraud_lift or float("nan")
            rel = abs(lift - ref_lift) / ref_lift if ref_lift == ref_lift else 1.0
            m.append(FidelityMetric(
                name="cnp_fraud_lift", family="stylised", value=lift,
                target=ref_lift, verdict=_verdict(rel, 0.35, 0.70),
                interpretation="CNP-to-CP fraud-rate ratio"))

    if is_fraud_col in gen:
        fr = float(gen[is_fraud_col].mean())
        rel = abs(fr - profile.fraud_rate) / max(profile.fraud_rate, 1e-9)
        m.append(FidelityMetric(
            name="fraud_base_rate", family="stylised", value=fr,
            target=profile.fraud_rate, verdict=_verdict(rel, 0.50, 2.00),
            interpretation="prevalence; drives PR-AUC comparability"))

    return m


# --------------------------------------------------------------------------
# Adversarial: the headline metric
# --------------------------------------------------------------------------


def discriminator_auc(
    gen: pd.DataFrame,
    ref: pd.DataFrame,
    features: List[str],
    n_splits: int = 4,
    seed: int = 0,
) -> float:
    """Train a classifier to separate generated from reference rows.

    Returned AUC is folded about 0.5 - a discriminator that is confidently
    *wrong* is just as much a fidelity failure as one that is confidently
    right, and unfolded AUC would hide that.
    """
    import lightgbm as lgb

    g = gen[features].copy()
    r = ref[features].copy()
    X = pd.concat([g, r], ignore_index=True)
    y = np.r_[np.ones(len(g)), np.zeros(len(r))]

    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")

    aucs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, subsample=0.9, colsample_bytree=0.9,
            random_state=seed, verbosity=-1,
        )
        clf.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X.iloc[te])[:, 1]))

    auc = float(np.mean(aucs))
    return max(auc, 1.0 - auc)


def evaluate(
    gen: pd.DataFrame,
    profile: CalibrationProfile,
    ref: Optional[pd.DataFrame] = None,
    discriminator_features: Optional[List[str]] = None,
    discriminator_rows: int = 300_000,
    seed: int = 1,
    **cols,
) -> FidelityReport:
    """Score `gen` against the calibration targets and the reference corpus.

    Distributional metrics are computed on the FULL generated frame; only the
    discriminator is subsampled. Scoring merchant concentration on a subsample
    under-measures it badly - the same generator reads alpha 1.95 on 1.6M rows
    and 1.63 on a 300k sample of those same rows, because thinning transactions
    per merchant flattens the rank-frequency slope.
    """
    metrics = score_against_profile(gen, profile, **cols)
    auc = None

    if ref is not None:
        amount_col = cols.get("amount_col", "amount")
        g = gen[amount_col].to_numpy(dtype=float)
        r = ref[amount_col].to_numpy(dtype=float)
        g, r = g[g > 0], r[r > 0]
        if len(g) and len(r):
            ks = float(stats.ks_2samp(g[:200_000], r[:200_000]).statistic)
            metrics.append(FidelityMetric(
                name="amount_ks_vs_reference", family="marginal", value=ks,
                target=0.0, verdict=_verdict(ks, 0.05, 0.15),
                interpretation="two-sample KS against the reference corpus itself"))

    if ref is not None and discriminator_features:
        n = min(len(gen), len(ref), discriminator_rows)
        g_s = gen.sample(n, random_state=seed) if len(gen) > n else gen
        r_s = ref.sample(n, random_state=seed) if len(ref) > n else ref
        auc = discriminator_auc(g_s, r_s, discriminator_features)
        metrics.append(FidelityMetric(
            name="discriminator_auc", family="adversarial", value=auc, target=0.5,
            verdict=_verdict(auc, 0.75, 0.90),
            interpretation="0.5 = indistinguishable from reference"))
    return FidelityReport(
        metrics=metrics, discriminator_auc=auc,
        n_generated=len(gen), n_reference=len(ref) if ref is not None else 0,
        reference_name=profile.name,
    )
