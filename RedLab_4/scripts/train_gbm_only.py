"""Supervised GBM baseline only - split from train_defend.py."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sklearn.metrics import average_precision_score, roc_auc_score

from redlab.defend.detect import Detector, recall_precision_at_fpr
from redlab.defend.features import build_features
from redlab.taxonomy.loader import Taxonomy
import pandas as pd

df = pd.read_parquet("data/processed/world_attacked.parquet")
tax = Taxonomy.load()
projected_ids = {v.id for v in tax if v.maturity.value == "projected"}
is_projected = df.attack_id.isin(projected_ids)

print("building features...", flush=True)
feats = build_features(df)
print(f"built {len(feats):,} rows", flush=True)

cut = feats.timestamp.quantile(0.7)
ip = is_projected.reindex(feats.index).to_numpy() if hasattr(is_projected, "reindex") else is_projected.values
train = feats[(feats.timestamp <= cut) & ~(feats.is_fraud.eq(1) & ip)]
test = feats[(feats.timestamp > cut) & (feats.is_fraud.eq(0) | (feats.is_fraud.eq(1) & ip))]
print(f"train {len(train):,} | test {len(test):,} ({int(test.is_fraud.sum())} fraud)",
     flush=True)

sup = Detector(n_estimators=300).fit(train)
print("GBM trained", flush=True)
score = sup.score(test)
y = test.is_fraud.to_numpy()
roc = roc_auc_score(y, score)
pr = average_precision_score(y, score)
rec, prec = recall_precision_at_fpr(y, score, 0.005)
print(f"GBM: ROC-AUC {roc:.4f}  PR-AUC {pr:.4f}  recall@0.5%FPR {rec*100:.1f}%  "
     f"precision {prec*100:.1f}%", flush=True)

json.dump({"supervised": {"roc_auc": roc, "pr_auc": pr, "recall_at_0.5fpr": rec,
                         "precision": prec}},
         open("artifacts/defend_gbm_eval.json", "w"), indent=1)
print("-> artifacts/defend_gbm_eval.json", flush=True)
