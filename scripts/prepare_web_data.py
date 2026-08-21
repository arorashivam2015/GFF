"""Prepare the artifacts the web prototype serves: a persisted detector and a
scored transaction sample for the Defense Console. Run after train_detector.py."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from redlab.defend.detect import Detector, recall_precision_at_fpr, temporal_split

F = pd.read_parquet("data/processed/features.parquet")
tr, te = temporal_split(F)
det = Detector(n_estimators=400).fit(tr)

p = det.score(te)
te = te.copy()
te["risk_score"] = p

neg = p[te.is_fraud.to_numpy() == 0]
thresh_review = float(pd.Series(neg).quantile(1 - 0.005))   # 0.5% FPR budget
thresh_block = float(pd.Series(neg).quantile(1 - 0.001))    # 0.1% FPR budget

def decide(row):
    if row.risk_score >= thresh_block:
        return "BLOCK"
    if row.risk_score >= thresh_review:
        return "STEP-UP"
    return "ALLOW"

te["decision"] = te.apply(decide, axis=1)

pathlib.Path("artifacts/model").mkdir(parents=True, exist_ok=True)
joblib.dump({"detector": det, "thresh_review": thresh_review,
            "thresh_block": thresh_block}, "artifacts/model/detector.joblib")

# A presentable console sample: all fraud the model catches or misses, plus a
# slice of legitimate traffic, ordered by time so it reads as a live feed.
fraud = te[te.is_fraud == 1]
legit_all = te[te.is_fraud == 0]

# Sample legit DENSELY enough that no time window is fraud-only. A first pass
# sampled only 4,000 of ~490k legit rows; fraud campaigns cluster tightly in
# time (see FINDINGS.md #6 - real fraud clusters per victim), so the tail of
# the time-sorted sample landed 100% fraud with no legit nearby, which made
# the default "live feed" view misrepresent the actual class balance. Sampling
# stratified by day keeps every day fraud touches represented on the legit
# side too.
legit_all = legit_all.assign(_day=legit_all.timestamp.dt.floor("D"))
per_day = max(int(60_000 / legit_all._day.nunique()), 20)
legit_sample = (legit_all.groupby("_day", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), per_day), random_state=1))
               .drop(columns="_day"))

console = pd.concat([fraud, legit_sample]).sort_values("timestamp")
cols = ["txn_id", "timestamp", "user_id", "merchant_id", "device_id", "mcc",
       "channel", "amount", "is_fraud", "attack_id", "risk_score", "decision"]
console[cols].to_parquet("artifacts/model/console_sample.parquet", index=False)

precision_review = float(((te.decision != "ALLOW") & (te.is_fraud == 1)).sum() /
                         max((te.decision != "ALLOW").sum(), 1))
recall_block, _ = recall_precision_at_fpr(te.is_fraud.to_numpy(), p, 0.001)

print(f"detector persisted -> artifacts/model/detector.joblib")
print(f"console sample      -> artifacts/model/console_sample.parquet "
      f"({len(console):,} rows, {int(console.is_fraud.sum()):,} fraud)")
print(f"thresholds: review>={thresh_review:.4f}  block>={thresh_block:.4f}")
print(f"decision mix: {console.decision.value_counts().to_dict()}")
