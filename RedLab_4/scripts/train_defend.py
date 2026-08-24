"""Train the primary unsupervised detector (autoencoder, zero fraud-label
exposure) and a supervised GBM comparison baseline, both evaluated on a
held-out mechanism neither has trained on - the taxonomy's "projected"
maturity vectors, same discipline RedLab_6 established for this exact
purpose."""
import json
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from redlab.defend.anomaly_ae import AnomalyDetector, featurize
from redlab.defend.detect import Detector, recall_precision_at_fpr
from redlab.defend.features import build_features, feature_names
from redlab.sim.cvae import VAEEncoding
from redlab.taxonomy.loader import Taxonomy

df = pd.read_parquet("data/processed/world_attacked.parquet")
tax = Taxonomy.load()
projected_ids = {v.id for v in tax if v.maturity.value == "projected"}
print(f"{len(projected_ids)} of {len(tax)} vectors rated 'projected' - held out of "
     f"BOTH detectors' training entirely")

is_projected = df.attack_id.isin(projected_ids)
cut = df.timestamp.quantile(0.7)
train = df[(df.timestamp <= cut) & ~(df.is_fraud.eq(1) & is_projected)]
test = df[(df.timestamp > cut) & (df.is_fraud.eq(0) | (df.is_fraud.eq(1) & is_projected))]
print(f"train: {len(train):,} rows, {int(train.is_fraud.sum()):,} fraud "
     f"(0 projected-vector fraud)")
print(f"test:  {len(test):,} rows, {int(test.is_fraud.sum()):,} fraud "
     f"(100% projected-vector or legitimate)")

# --- unsupervised autoencoder, trained on legit rows only -------------------
enc = pickle.load(open("artifacts/model/vae_encoding.pkl", "rb"))
legit_train = train[train.is_fraud == 0]
legit_x = featurize(legit_train, enc.mcc_to_idx, enc.chan_to_idx,
                    enc.amount_log_mean, enc.amount_log_std)
in_dim = legit_x.shape[1]
print(f"\nfeaturized legit training rows: {legit_x.shape}")

det = AnomalyDetector(in_dim=in_dim, epochs=15, seed=0).fit(legit_x)
print("autoencoder trained")

test_x = featurize(test, enc.mcc_to_idx, enc.chan_to_idx,
                   enc.amount_log_mean, enc.amount_log_std)
score_ae = det.score(test_x)
y = test.is_fraud.to_numpy()
roc_ae = roc_auc_score(y, score_ae)
pr_ae = average_precision_score(y, score_ae)
rec_ae, prec_ae = recall_precision_at_fpr(y, score_ae, 0.005)
print(f"\nAUTOENCODER (zero fraud-label exposure)")
print(f"  ROC-AUC {roc_ae:.4f}  PR-AUC {pr_ae:.4f}  recall@0.5%FPR {rec_ae*100:.1f}%  "
     f"precision {prec_ae*100:.1f}%")

# --- supervised GBM baseline, trained on observed+emerging fraud only ------
feats = build_features(df)
feats_train = feats[(feats.timestamp <= cut) & ~(feats.is_fraud.eq(1) & is_projected.values)]
feats_test = feats[(feats.timestamp > cut) &
                   (feats.is_fraud.eq(0) | (feats.is_fraud.eq(1) & is_projected.values))]
sup = Detector(n_estimators=300).fit(feats_train)
score_sup = sup.score(feats_test)
roc_sup = roc_auc_score(y, score_sup)
pr_sup = average_precision_score(y, score_sup)
rec_sup, prec_sup = recall_precision_at_fpr(y, score_sup, 0.005)
print(f"\nSUPERVISED BASELINE (observed+emerging fraud only)")
print(f"  ROC-AUC {roc_sup:.4f}  PR-AUC {pr_sup:.4f}  recall@0.5%FPR {rec_sup*100:.1f}%  "
     f"precision {prec_sup*100:.1f}%")

print(f"\n{'='*66}\nHEADLINE: recall@0.5%FPR on NEVER-TRAINED-ON (projected) attacks")
print(f"{'='*66}")
print(f"  autoencoder (zero labels): {rec_ae*100:5.1f}%")
print(f"  supervised baseline:       {rec_sup*100:5.1f}%")

torch.save(det.model.state_dict(), "artifacts/model/anomaly_ae.pt")
pathlib.Path("artifacts").mkdir(exist_ok=True)
json.dump({
    "n_projected_vectors": len(projected_ids),
    "autoencoder": {"roc_auc": roc_ae, "pr_auc": pr_ae, "recall_at_0.5fpr": rec_ae,
                    "precision": prec_ae, "threshold": det.threshold},
    "supervised": {"roc_auc": roc_sup, "pr_auc": pr_sup, "recall_at_0.5fpr": rec_sup,
                  "precision": prec_sup},
}, open("artifacts/defend_eval.json", "w"), indent=1)
print("\n-> artifacts/defend_eval.json, artifacts/model/anomaly_ae.pt")
