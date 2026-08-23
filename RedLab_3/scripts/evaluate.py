"""Headline comparison: unsupervised ensemble (zero fraud-label exposure)
vs. supervised GBM baseline, both scored on the zero-exposure holdout."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from redlab.defend.anomaly import AnomalyEnsemble
from redlab.defend.detect import Detector, recall_precision_at_fpr
from redlab.defend.features import feature_names
from redlab.defend.zero_exposure import verify_no_leakage, zero_exposure_split
from redlab.taxonomy.loader import Taxonomy

F = pd.read_parquet("data/processed/features.parquet")
tax = Taxonomy.load()
n_projected = sum(1 for v in tax if v.maturity.value == "projected")
print(f"taxonomy: {len(tax)} vectors, {n_projected} rated 'projected' (never in training)")

train, test = zero_exposure_split(F, tax)
verify_no_leakage(train, tax)
print(f"train: {len(train):,} rows, {int(train.is_fraud.sum()):,} fraud "
     f"(0 projected-vector fraud, verified)")
print(f"test:  {len(test):,} rows, {int(test.is_fraud.sum()):,} fraud "
     f"(100% projected-vector or legitimate)")

cols = feature_names(F)

# --- unsupervised ensemble, trained on legit rows only ----------------------
legit_train = train[train.is_fraud == 0]
rng = np.random.default_rng(0)
holdout_idx = rng.choice(len(legit_train), size=min(20000, len(legit_train) // 4), replace=False)
legit_for_fit = legit_train.drop(legit_train.index[holdout_idx])
legit_coverage_holdout = legit_train.iloc[holdout_idx]

ens = AnomalyEnsemble(target_fpr=0.005, seed=0).fit(legit_for_fit, cols)
coverage = ens.verify_coverage(legit_coverage_holdout)
print(f"\nconformal coverage check: target FPR 0.500%, empirical FPR on held-out "
     f"legit rows = {coverage*100:.3f}%")

score_ens = ens.score(test)
y = test.is_fraud.to_numpy()
roc_ens = roc_auc_score(y, score_ens)
pr_ens = average_precision_score(y, score_ens)
rec_ens, prec_ens = recall_precision_at_fpr(y, score_ens, 0.005)
print(f"\nUNSUPERVISED ENSEMBLE (zero fraud-label exposure, ever)")
print(f"  ROC-AUC {roc_ens:.4f}  PR-AUC {pr_ens:.4f}  "
     f"recall@0.5%FPR {rec_ens*100:.1f}%  precision {prec_ens*100:.1f}%")

# --- supervised baseline, trained on observed+emerging fraud only ----------
det = Detector(n_estimators=300).fit(train)
score_sup = det.score(test)
roc_sup = roc_auc_score(y, score_sup)
pr_sup = average_precision_score(y, score_sup)
rec_sup, prec_sup = recall_precision_at_fpr(y, score_sup, 0.005)
print(f"\nSUPERVISED BASELINE (trained on observed+emerging fraud, never projected)")
print(f"  ROC-AUC {roc_sup:.4f}  PR-AUC {pr_sup:.4f}  "
     f"recall@0.5%FPR {rec_sup*100:.1f}%  precision {prec_sup*100:.1f}%")

print(f"\n{'='*66}\nHEADLINE: recall@0.5%FPR on NEVER-TRAINED-ON attacks")
print(f"{'='*66}")
print(f"  unsupervised ensemble:  {rec_ens*100:5.1f}%")
print(f"  supervised baseline:    {rec_sup*100:5.1f}%")

pathlib.Path("artifacts").mkdir(exist_ok=True)
json.dump({
    "n_projected_vectors": n_projected,
    "conformal_target_fpr": 0.005,
    "conformal_empirical_fpr": coverage,
    "unsupervised": {"roc_auc": roc_ens, "pr_auc": pr_ens, "recall_at_0.5fpr": rec_ens,
                     "precision": prec_ens},
    "supervised": {"roc_auc": roc_sup, "pr_auc": pr_sup, "recall_at_0.5fpr": rec_sup,
                  "precision": prec_sup},
}, open("artifacts/eval_results.json", "w"), indent=1)
print("\n-> artifacts/eval_results.json")
