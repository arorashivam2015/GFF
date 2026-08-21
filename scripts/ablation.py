"""Feature-group ablation and mechanism-based holdout.

Runs on a stratified subsample: 13 model fits on the full 1.6M frame is not a
useful use of wall-clock, and ablation compares models against each other, so a
consistent subsample preserves the comparison.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from redlab.defend.detect import Detector
from redlab.taxonomy.loader import Taxonomy

META = ["txn_id", "timestamp", "user_id", "merchant_id", "device_id",
        "is_fraud", "attack_id"]
GROUPS = {
    "raw": ["amount", "abs_amount", "is_refund", "hour", "dow", "is_online",
            "mcc", "channel"],
    "baseline": ["u_prior_n", "u_amt_over_mean", "u_amt_over_max", "u_amt_z",
                 "u_exceeds_prior_max"],
    "velocity": ["u_secs_since_last", "u_txn_1h", "u_txn_24h", "u_txn_7d",
                 "m_txn_1h", "d_txn_24h"],
    "novelty": ["u_first_merchant", "u_first_mcc", "u_first_device",
                "u_first_state", "u_distinct_merch_prior", "u_merch_share"],
    "sharing": ["d_distinct_users_prior", "m_distinct_users_prior",
                "m_prior_n", "d_prior_n"],
}

F = pd.read_parquet("data/processed/features.parquet")
cut = F.timestamp.quantile(0.70)

# Stratified subsample: keep all fraud, thin the negatives.
fr = F[F.is_fraud == 1]
lg = F[F.is_fraud == 0].sample(400_000, random_state=0)
S = pd.concat([fr, lg]).sort_values("timestamp").reset_index(drop=True)
tr, te = S[S.timestamp <= cut], S[S.timestamp > cut]
ALL = [c for c in F.columns if c not in META]
print(f"subsample {len(S):,} rows | train {len(tr):,} | test {len(te):,} "
      f"| test fraud {int(te.is_fraud.sum()):,}\n")

D = dict(n_estimators=250)
rows = []


def run(label, cols):
    d = Detector(**D).fit(tr[META + cols])
    r = d.evaluate(te[META + cols], label)
    rows.append((label, r.pr_auc, r.recall_at_fpr["0.5%"], r.n_positive))
    print(f"{label:<26}{r.pr_auc:>9.4f}{r.recall_at_fpr['0.5%']*100:>9.1f}%")


print(f"{'feature set':<26}{'PR-AUC':>9}{'rec@.5%':>10}")
run("ALL", ALL)
for name, cols in GROUPS.items():
    run(f"only {name}", [c for c in cols if c in F.columns])
for name, cols in GROUPS.items():
    run(f"all minus {name}", [c for c in ALL if c not in cols])

# --- mechanism holdout -------------------------------------------------
print("\nMECHANISM HOLDOUT (generative axes, not family labels)")
print(f"{'holdout':<44}{'PR-AUC':>9}{'rec@.5%':>10}{'n_pos':>9}")
tax = Taxonomy.load()
prof = {v.id: (v.simulation.amount_profile.value, v.simulation.temporal_shape.value)
        for v in tax}
S["_amt"] = S.attack_id.map(lambda a: prof.get(a, ("", ""))[0] if pd.notna(a) else "")
S["_tmp"] = S.attack_id.map(lambda a: prof.get(a, ("", ""))[1] if pd.notna(a) else "")

mech = []
for axis, vals in [("_amt", ["micro_probe", "drain"]),
                   ("_amt", ["typical", "just_under_limit"]),
                   ("_tmp", ["slow_drip", "dormant_then_spike"]),
                   ("_tmp", ["burst"])]:
    hold = S[axis].isin(vals)
    t1 = S[(S.timestamp <= cut) & ~(S.is_fraud.eq(1) & hold)]
    t2 = S[(S.timestamp > cut) & (S.is_fraud.eq(0) | hold)]
    if t2.is_fraud.sum() < 30:
        continue
    d = Detector(**D).fit(t1[META + ALL])
    r = d.evaluate(t2[META + ALL], f"{axis[1:]}={vals}")
    mech.append((f"{axis[1:]}={'+'.join(vals)}", r.pr_auc,
                 r.recall_at_fpr["0.5%"], r.n_positive))
    print(f"{axis[1:]+'='+'+'.join(vals):<44}{r.pr_auc:>9.4f}"
          f"{r.recall_at_fpr['0.5%']*100:>9.1f}%{r.n_positive:>9,}")

json.dump({"ablation": rows, "mechanism_holdout": mech},
          open("artifacts/ablation.json", "w"), indent=1)
print("\n-> artifacts/ablation.json")
