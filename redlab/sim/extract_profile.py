"""Extract a CalibrationProfile from the TabFormer reference corpus.

    python -m redlab.sim.extract_profile \
        --csv data/raw/card_transaction.v1.csv \
        --out data/processed/reference_profile.json

Single streaming pass; inter-transaction timing is computed on a user sample
because it needs per-user sorting and the full join is not worth the memory.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import (
    AmountTargets,
    CalibrationProfile,
    Confidence,
    MerchantTargets,
    TemporalTargets,
    benford_mad,
    fit_zipf_alpha,
)

PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
USECOLS = ["User", "Card", "Year", "Month", "Day", "Time", "Amount",
           "Use Chip", "Merchant Name", "Merchant State", "MCC",
           "Errors?", "Is Fraud?"]

PROVENANCE = (
    "IBM TabFormer credit-card transaction dataset (Padhi et al., ICASSP 2021), "
    "github.com/IBM/TabFormer. 24.4M records. IBM documents this corpus as "
    "SYNTHETIC - it is used here as an externally-authored reference for "
    "distributional comparison, NOT as real payment data."
)


def extract(csv: Path, sample_users: int = 250, chunksize: int = 2_000_000) -> CalibrationProfile:
    n = fraud = neg = zero = 0
    amt_pool = []
    mcc, mcc_fraud = Counter(), Counter()
    chip, chip_fraud = Counter(), Counter()
    err, hour, dow = Counter(), Counter(), Counter()
    user_ct, merch = Counter(), Counter()
    first_digit = Counter()
    inter_pool = []

    sampled = set(range(sample_users))

    for ch in pd.read_csv(csv, usecols=USECOLS, chunksize=chunksize,
                          dtype={"Merchant Name": str, "MCC": "int32", "User": "int32"}):
        amt = ch["Amount"].str.replace("$", "", regex=False).astype("float64")
        isf = ch["Is Fraud?"] == "Yes"

        n += len(ch)
        fraud += int(isf.sum())
        neg += int((amt < 0).sum())
        zero += int((amt == 0).sum())

        amt_pool.append(amt.sample(min(300_000, len(amt)), random_state=0).to_numpy())
        mcc.update(ch["MCC"].tolist())
        mcc_fraud.update(ch.loc[isf, "MCC"].tolist())
        chip.update(ch["Use Chip"].tolist())
        chip_fraud.update(ch.loc[isf, "Use Chip"].tolist())
        err.update(ch["Errors?"].fillna("(none)").tolist())
        user_ct.update(ch["User"].tolist())
        merch.update(ch["Merchant Name"].tolist())

        hh = ch["Time"].str.slice(0, 2).astype("int16")
        hour.update(hh.tolist())

        ts = pd.to_datetime(
            dict(year=ch["Year"], month=ch["Month"], day=ch["Day"]), errors="coerce"
        )
        dow.update(ts.dt.dayofweek.dropna().astype("int8").tolist())

        pos = amt[amt > 0]
        first_digit.update(
            pos.map(lambda x: f"{x:.10f}".lstrip("0.").lstrip("0")[:1]).tolist()
        )

        sub = ch[ch["User"].isin(sampled)]
        if len(sub):
            t = pd.to_datetime(
                dict(year=sub["Year"], month=sub["Month"], day=sub["Day"]), errors="coerce"
            ) + pd.to_timedelta(sub["Time"] + ":00")
            g = pd.DataFrame({"u": sub["User"].values, "t": t.values}).dropna()
            g = g.sort_values(["u", "t"])
            d = g.groupby("u")["t"].diff().dt.total_seconds() / 3600.0
            inter_pool.append(d.dropna().to_numpy())

    A = np.concatenate(amt_pool)
    pos_amt = A[A > 0]
    fd = np.array([first_digit.get(str(d), 0) for d in range(1, 10)], dtype=float)
    fd = fd / fd.sum()
    logs = np.log(pos_amt[pos_amt > 0])
    inter = np.concatenate(inter_pool) if inter_pool else np.array([np.nan])
    merch_counts = np.array(list(merch.values()), dtype=float)
    mc_sorted = np.sort(merch_counts)[::-1]
    top1 = mc_sorted[: max(1, len(mc_sorted) // 100)].sum() / mc_sorted.sum()
    uc = np.array(list(user_ct.values()), dtype=float)

    profile = CalibrationProfile(
        name="tabformer_card_reference",
        provenance=PROVENANCE,
        confidence=Confidence.REFERENCE_CORPUS,
        n_rows=n,
        n_users=len(user_ct),
        fraud_rate=fraud / n,
        amount=AmountTargets(
            percentiles={f"p{p}": float(np.percentile(pos_amt, p)) for p in PCTS},
            lognormal_mu=float(logs.mean()),
            lognormal_sigma=float(logs.std()),
            mean=float(pos_amt.mean()),
            benford_first_digit=[float(x) for x in fd],
            refund_share=neg / n,
            zero_share=zero / n,
        ),
        merchant=MerchantTargets(
            n_merchants=len(merch),
            top1pct_volume_share=float(top1),
            zipf_alpha=fit_zipf_alpha(merch_counts),
            mcc_mix={str(k): v / n for k, v in mcc.most_common(80)},
            mcc_fraud_rate={str(k): mcc_fraud.get(k, 0) / v for k, v in mcc.most_common(80)},
        ),
        temporal=TemporalTargets(
            hour_of_day=[hour.get(h, 0) / n for h in range(24)],
            day_of_week=[dow.get(d, 0) / max(sum(dow.values()), 1) for d in range(7)],
            inter_txn_hours={f"p{p}": float(np.nanpercentile(inter, p)) for p in PCTS},
        ),
        channel_mix={str(k): v / n for k, v in chip.items()},
        channel_fraud_rate={str(k): chip_fraud.get(k, 0) / v for k, v in chip.items()},
        error_mix={str(k): v / n for k, v in err.most_common(12)},
        per_user_txn_percentiles={f"p{p}": float(np.percentile(uc, p)) for p in PCTS},
        notes=f"benford_mad_pp={benford_mad(fd):.3f}; "
              f"inter-txn timing from {sample_users}-user sample",
    )
    return profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/card_transaction.v1.csv")
    ap.add_argument("--out", default="data/processed/reference_profile.json")
    ap.add_argument("--sample-users", type=int, default=250)
    a = ap.parse_args()

    prof = extract(Path(a.csv), sample_users=a.sample_users)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(prof.model_dump(mode="json"), indent=1))

    print(f"wrote {a.out}")
    print(f"  rows            {prof.n_rows:,}")
    print(f"  users           {prof.n_users:,}")
    print(f"  fraud rate      {prof.fraud_rate*100:.4f}%")
    print(f"  CNP fraud lift  {prof.cnp_fraud_lift:.1f}x")
    print(f"  amount lognorm  mu={prof.amount.lognormal_mu:.3f} "
          f"sigma={prof.amount.lognormal_sigma:.3f}")
    print(f"  zipf alpha      {prof.merchant.zipf_alpha:.3f}")
    print(f"  top1% share     {prof.merchant.top1pct_volume_share*100:.1f}%")
    print(f"  benford MAD     {benford_mad(np.array(prof.amount.benford_first_digit)):.3f} pp")
    print(f"  inter-txn med   {prof.temporal.inter_txn_hours['p50']:.2f} h")


if __name__ == "__main__":
    main()
