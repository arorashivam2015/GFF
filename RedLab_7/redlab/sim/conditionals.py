"""Conditional structure extracted from the reference corpus.

Marginal targets alone produced discriminator AUC 0.847 in harness validation:
a generator can match every marginal almost exactly and stay trivially
separable, because real payment data is defined by its *joint* structure -
amount depends on category, category on hour, channel on category, merchant on
the cardholder's geography and habits.

This module captures those conditionals so the simulator can reproduce them
rather than inventing them.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class MccProfile(BaseModel):
    """Everything the simulator needs to emit a transaction in one category."""

    model_config = ConfigDict(extra="forbid")

    mcc: str
    share: float = Field(..., description="share of all transactions")
    amount_mu: float = Field(..., description="log-normal mu of amount | mcc")
    amount_sigma: float
    amount_median: float
    amount_quantiles: List[float] = Field(
        ...,
        min_length=101,
        max_length=101,
        description="empirical inverse CDF of amount | mcc, at q=0.00..1.00. "
        "Sampled through rather than fitted: reference log-amounts carry skew "
        "-0.71 and a truncated upper tail, so a log-normal fit matches the "
        "variance but overshoots p99 by 2-8x.",
    )
    round_share: float = Field(
        ..., ge=0, le=1, description="share of amounts landing on a whole unit"
    )
    hour_dist: List[float] = Field(..., min_length=24, max_length=24)
    channel_dist: Dict[str, float]
    fraud_rate: float
    n_merchants: int
    refund_share: float


class UserProfile(BaseModel):
    """Population-level distributions the simulator samples archetypes from."""

    model_config = ConfigDict(extra="forbid")

    txns_per_day: Dict[str, float] = Field(..., description="pXX of per-user daily rate")
    distinct_merchants: Dict[str, float]
    merchant_loyalty: Dict[str, float] = Field(
        ..., description="pXX of share-of-txns at a user's top-5 merchants"
    )
    distinct_mccs: Dict[str, float]
    home_state_share: Dict[str, float] = Field(
        ..., description="pXX of share-of-txns in the user's modal state"
    )


class ConditionalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provenance: str
    mccs: Dict[str, MccProfile]
    users: UserProfile
    state_mix: Dict[str, float]
    merchant_mcc_exclusivity: float = Field(
        ..., description="share of merchants trading under exactly one MCC"
    )
    online_state_token: str = "ONLINE"

    def mcc_keys(self) -> List[str]:
        return list(self.mccs)

    def mcc_weights(self) -> np.ndarray:
        w = np.array([m.share for m in self.mccs.values()], dtype=float)
        return w / w.sum()

    @classmethod
    def load(cls, path: str) -> "ConditionalProfile":
        return cls.model_validate(json.loads(Path(path).read_text()))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.model_dump(mode="json"), indent=1))


PCTS = [5, 10, 25, 50, 75, 90, 95, 99]
USECOLS = ["User", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
           "Merchant Name", "Merchant State", "MCC", "Is Fraud?"]


def extract(csv: Path, top_mccs: int = 60, user_sample: int = 400,
            chunksize: int = 2_000_000) -> ConditionalProfile:
    n = 0
    mcc_amt: Dict[str, List[np.ndarray]] = defaultdict(list)
    mcc_hour: Dict[str, Counter] = defaultdict(Counter)
    mcc_chan: Dict[str, Counter] = defaultdict(Counter)
    mcc_n = Counter()
    mcc_fraud = Counter()
    mcc_neg = Counter()
    mcc_merch: Dict[str, set] = defaultdict(set)
    merch_mcc: Dict[str, set] = defaultdict(set)
    state = Counter()

    sampled = set(range(user_sample))
    u_merch: Dict[int, Counter] = defaultdict(Counter)
    u_mcc: Dict[int, Counter] = defaultdict(Counter)
    u_state: Dict[int, Counter] = defaultdict(Counter)
    u_days: Dict[int, set] = defaultdict(set)
    u_n = Counter()

    for ch in pd.read_csv(csv, usecols=USECOLS, chunksize=chunksize,
                          dtype={"Merchant Name": str, "MCC": "int32", "User": "int32"}):
        amt = ch["Amount"].str.replace("$", "", regex=False).astype("float64")
        mcc = ch["MCC"].astype(str)
        isf = ch["Is Fraud?"] == "Yes"
        hh = ch["Time"].str.slice(0, 2).astype("int16")
        st = ch["Merchant State"].fillna("ONLINE").astype(str)
        n += len(ch)

        mcc_n.update(mcc.tolist())
        mcc_fraud.update(mcc[isf].tolist())
        mcc_neg.update(mcc[amt < 0].tolist())
        state.update(st.tolist())

        pos = amt > 0
        for k, grp in pd.DataFrame({"m": mcc[pos], "a": amt[pos]}).groupby("m", sort=False):
            mcc_amt[k].append(grp["a"].sample(min(30_000, len(grp)), random_state=0).to_numpy())
        for k, grp in pd.DataFrame({"m": mcc, "h": hh}).groupby("m", sort=False):
            mcc_hour[k].update(grp["h"].tolist())
        for k, grp in pd.DataFrame({"m": mcc, "c": ch["Use Chip"]}).groupby("m", sort=False):
            mcc_chan[k].update(grp["c"].tolist())
        for k, grp in pd.DataFrame({"m": mcc, "n": ch["Merchant Name"]}).groupby("m", sort=False):
            mcc_merch[k].update(grp["n"].unique().tolist())
        for k, grp in pd.DataFrame({"n": ch["Merchant Name"], "m": mcc}).groupby("n", sort=False):
            merch_mcc[k].update(grp["m"].unique().tolist())

        sub = ch[ch["User"].isin(sampled)]
        if len(sub):
            su_mcc = mcc[sub.index]
            su_st = st[sub.index]
            for u, m, nm, s, y, mo, d in zip(sub["User"], su_mcc, sub["Merchant Name"],
                                             su_st, sub["Year"], sub["Month"], sub["Day"]):
                u_merch[u][nm] += 1
                u_mcc[u][m] += 1
                u_state[u][s] += 1
                u_days[u].add((y, mo, d))
                u_n[u] += 1

    keep = [k for k, _ in mcc_n.most_common(top_mccs)]
    mccs: Dict[str, MccProfile] = {}
    for k in keep:
        a = np.concatenate(mcc_amt[k]) if mcc_amt[k] else np.array([1.0])
        logs = np.log(a[a > 0])
        hc = np.array([mcc_hour[k].get(h, 0) for h in range(24)], dtype=float)
        cc = mcc_chan[k]
        tot_c = sum(cc.values()) or 1
        cents = np.round((a - np.floor(a)) * 100).astype(int) % 100
        mccs[k] = MccProfile(
            mcc=k,
            share=mcc_n[k] / n,
            amount_mu=float(logs.mean()),
            amount_sigma=float(logs.std()),
            amount_median=float(np.median(a)),
            amount_quantiles=[float(x) for x in np.percentile(a, np.linspace(0, 100, 101))],
            round_share=float(np.mean(cents == 0)),
            hour_dist=list(hc / hc.sum()) if hc.sum() else [1 / 24] * 24,
            channel_dist={str(c): v / tot_c for c, v in cc.items()},
            fraud_rate=mcc_fraud.get(k, 0) / mcc_n[k],
            n_merchants=len(mcc_merch[k]),
            refund_share=mcc_neg.get(k, 0) / mcc_n[k],
        )

    def pct(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=float)
        return {f"p{p}": float(np.percentile(arr, p)) for p in PCTS} if len(arr) else {}

    users = list(u_n)
    loyalty = [sum(c for _, c in u_merch[u].most_common(5)) / u_n[u] for u in users]
    rate = [u_n[u] / max(len(u_days[u]), 1) for u in users]
    home = [u_state[u].most_common(1)[0][1] / u_n[u] for u in users]

    excl = sum(1 for v in merch_mcc.values() if len(v) == 1) / max(len(merch_mcc), 1)

    return ConditionalProfile(
        provenance="TabFormer reference corpus (synthetic; see calibration.py)",
        mccs=mccs,
        users=UserProfile(
            txns_per_day=pct(rate),
            distinct_merchants=pct([len(u_merch[u]) for u in users]),
            merchant_loyalty=pct(loyalty),
            distinct_mccs=pct([len(u_mcc[u]) for u in users]),
            home_state_share=pct(home),
        ),
        state_mix={k: v / n for k, v in state.most_common(60)},
        merchant_mcc_exclusivity=float(excl),
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/card_transaction.v1.csv")
    ap.add_argument("--out", default="data/processed/conditional_profile.json")
    a = ap.parse_args()

    p = extract(Path(a.csv))
    p.save(a.out)
    print(f"wrote {a.out}")
    print(f"  MCC profiles           {len(p.mccs)}")
    print(f"  merchant MCC exclusive {p.merchant_mcc_exclusivity*100:.1f}%")
    print(f"  user loyalty (top5)    p50={p.users.merchant_loyalty['p50']:.3f}")
    print(f"  txns/active-day        p50={p.users.txns_per_day['p50']:.2f}")
    print(f"  home-state share       p50={p.users.home_state_share['p50']:.3f}")
    print("\n  sample MCC conditionals (amount median | hour peak | online share):")
    for k in list(p.mccs)[:8]:
        m = p.mccs[k]
        peak = int(np.argmax(m.hour_dist))
        onl = sum(v for c, v in m.channel_dist.items() if "Online" in c)
        print(f"    {k:6s} ${m.amount_median:7.2f}   {peak:02d}:00   {onl*100:5.1f}%  "
              f"fraud={m.fraud_rate*100:.3f}%")
