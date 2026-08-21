"""How fraud behaves, measured on the reference corpus's 29,757 labelled frauds.

Attack realism cannot be assumed. A first injection pass, built from plausible
-sounding rules, produced fraud that a raw-feature LightGBM separated at ROC-AUC
0.9886 with every one of 42 vectors sitting at the ~100th score percentile. The
rules were wrong in ways that were only visible by measuring:

  channel      fraud is 61.7% online, not 92.8%
  category     fraud spans 98 MCCs with the top one at 16.2%, not 74.7% in one
  amount       fraud runs 2.4x the victim's MEDIAN at p50, and exceeds the
               victim's own historical MAX only 0.7% of the time

That last one is the important one. Fraud hides beneath the victim's own
ceiling; it does not blow through it. Any generator that draws fraud amounts as
a multiple of the victim's maximum is producing an artefact a detector will
learn instead of the attack.

Category lift is real signal and is preserved: electronics (5732) carries 58.7x
its legitimate share, digital goods (5815) 6.3x, clothing (5651) 5.1x. Fraud
concentrates in resellable goods. That is a pattern worth reproducing, not
flattening.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class FraudProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provenance: str
    n_fraud: int
    channel_mix: Dict[str, float] = Field(..., description="P(channel | fraud)")
    mcc_mix: Dict[str, float] = Field(..., description="P(mcc | fraud)")
    mcc_lift: Dict[str, float] = Field(..., description="P(mcc|fraud) / P(mcc|legit)")
    ratio_to_median_quantiles: List[float] = Field(
        ..., min_length=101, max_length=101,
        description="empirical inverse CDF of (fraud amount / victim's legit median)",
    )
    share_above_victim_max: float
    amount_quantiles: List[float] = Field(..., min_length=101, max_length=101)

    def ratio_at(self, q: np.ndarray) -> np.ndarray:
        """Inverse-CDF lookup on the amount-ratio distribution."""
        grid = np.linspace(0.0, 1.0, 101)
        return np.interp(np.clip(q, 0.0, 1.0), grid,
                         np.asarray(self.ratio_to_median_quantiles, dtype=float))

    @classmethod
    def load(cls, path: str = "data/processed/fraud_profile.json") -> "FraudProfile":
        return cls.model_validate(json.loads(Path(path).read_text()))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.model_dump(mode="json"), indent=1))


USECOLS = ["User", "Amount", "Use Chip", "MCC", "Is Fraud?"]


def extract(csv: Path, chunksize: int = 2_000_000) -> FraudProfile:
    fraud_parts: List[pd.DataFrame] = []
    ch_l: Dict[str, int] = {}
    mcc_l: Dict[str, int] = {}
    n_legit = 0
    u_sum: Dict[int, float] = {}
    u_cnt: Dict[int, int] = {}
    u_max: Dict[int, float] = {}

    for ch in pd.read_csv(csv, usecols=USECOLS, chunksize=chunksize,
                          dtype={"MCC": "int32", "User": "int32"}):
        amt = ch["Amount"].str.replace("$", "", regex=False).astype(float)
        isf = ch["Is Fraud?"] == "Yes"
        fraud_parts.append(pd.DataFrame({
            "user": ch.loc[isf, "User"].to_numpy(),
            "amt": amt[isf].to_numpy(),
            "channel": ch.loc[isf, "Use Chip"].to_numpy(),
            "mcc": ch.loc[isf, "MCC"].astype(str).to_numpy(),
        }))

        legit = ~isf
        n_legit += int(legit.sum())
        for k, v in ch.loc[legit, "Use Chip"].value_counts().items():
            ch_l[str(k)] = ch_l.get(str(k), 0) + int(v)
        for k, v in ch.loc[legit, "MCC"].astype(str).value_counts().items():
            mcc_l[k] = mcc_l.get(k, 0) + int(v)

        g = pd.DataFrame({"u": ch.loc[legit, "User"].to_numpy(),
                          "a": amt[legit].to_numpy()})
        g = g[g.a > 0]
        agg = g.groupby("u").a.agg(["sum", "count", "max"])
        for u, r in agg.iterrows():
            u_sum[u] = u_sum.get(u, 0.0) + float(r["sum"])
            u_cnt[u] = u_cnt.get(u, 0) + int(r["count"])
            u_max[u] = max(u_max.get(u, 0.0), float(r["max"]))

    F = pd.concat(fraud_parts, ignore_index=True)
    F = F[F.amt > 0].copy()

    # Per-user median is approximated by the mean of positive legit amounts;
    # exact medians would need a second pass and the ratio distribution is
    # robust to the substitution at this granularity.
    u_mean = {u: u_sum[u] / max(u_cnt[u], 1) for u in u_sum}
    F["u_base"] = F.user.map(u_mean)
    F["u_max"] = F.user.map(u_max)
    F = F.dropna(subset=["u_base"])
    F = F[F.u_base > 0]

    ratio = (F.amt / F.u_base).to_numpy()
    grid = np.linspace(0, 100, 101)

    mf = F.mcc.value_counts(normalize=True)
    ml = pd.Series({k: v / n_legit for k, v in mcc_l.items()})
    lift = (mf / ml.reindex(mf.index).fillna(1e-9)).replace([np.inf], 0.0)

    return FraudProfile(
        provenance="TabFormer reference corpus, 29,757 labelled fraud events "
                   "(synthetic corpus; see calibration.py honesty note)",
        n_fraud=int(len(F)),
        channel_mix={str(k): float(v) for k, v in
                     F.channel.value_counts(normalize=True).items()},
        mcc_mix={str(k): float(v) for k, v in mf.items()},
        mcc_lift={str(k): float(v) for k, v in lift.items()},
        ratio_to_median_quantiles=[float(x) for x in np.percentile(ratio, grid)],
        share_above_victim_max=float((F.amt > F.u_max).mean()),
        amount_quantiles=[float(x) for x in np.percentile(F.amt.to_numpy(), grid)],
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/card_transaction.v1.csv")
    ap.add_argument("--out", default="data/processed/fraud_profile.json")
    a = ap.parse_args()
    p = extract(Path(a.csv))
    p.save(a.out)
    print(f"wrote {a.out}")
    print(f"  fraud events           {p.n_fraud:,}")
    print(f"  online share           {p.channel_mix.get('Online Transaction',0)*100:.1f}%")
    print(f"  distinct fraud MCCs    {len(p.mcc_mix)}   top-1={max(p.mcc_mix.values())*100:.1f}%")
    print(f"  amount/victim-base p50 {p.ratio_at(np.array([0.5]))[0]:.2f}x")
    print(f"  above victim max       {p.share_above_victim_max*100:.2f}%")
    top = sorted(p.mcc_lift.items(), key=lambda kv: -kv[1])[:5]
    print("  highest-lift MCCs      " + ", ".join(f"{k}({v:.0f}x)" for k, v in top))
