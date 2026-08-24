"""Autoencoder training only - split from train_defend.py so a hang in one
piece doesn't lose the other's output. Saves its own artifact immediately."""
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
from redlab.defend.detect import recall_precision_at_fpr
from redlab.taxonomy.loader import Taxonomy

df = pd.read_parquet("data/processed/world_attacked.parquet")
tax = Taxonomy.load()
projected_ids = {v.id for v in tax if v.maturity.value == "projected"}
is_projected = df.attack_id.isin(projected_ids)
cut = df.timestamp.quantile(0.7)
train = df[(df.timestamp <= cut) & ~(df.is_fraud.eq(1) & is_projected)]
test = df[(df.timestamp > cut) & (df.is_fraud.eq(0) | (df.is_fraud.eq(1) & is_projected))]
print(f"train {len(train):,} | test {len(test):,} ({int(test.is_fraud.sum())} fraud)",
     flush=True)

enc = pickle.load(open("artifacts/model/vae_encoding.pkl", "rb"))
legit_train = train[train.is_fraud == 0]
legit_x = featurize(legit_train, enc.mcc_to_idx, enc.chan_to_idx,
                    enc.amount_log_mean, enc.amount_log_std)
print(f"featurized: {legit_x.shape}", flush=True)

det = AnomalyDetector(in_dim=legit_x.shape[1], epochs=15, seed=0).fit(legit_x)
print("trained", flush=True)

test_x = featurize(test, enc.mcc_to_idx, enc.chan_to_idx,
                   enc.amount_log_mean, enc.amount_log_std)
score = det.score(test_x)
y = test.is_fraud.to_numpy()
roc = roc_auc_score(y, score)
pr = average_precision_score(y, score)
rec, prec = recall_precision_at_fpr(y, score, 0.005)
print(f"AE: ROC-AUC {roc:.4f}  PR-AUC {pr:.4f}  recall@0.5%FPR {rec*100:.1f}%  "
     f"precision {prec*100:.1f}%", flush=True)

torch.save(det.model.state_dict(), "artifacts/model/anomaly_ae.pt")
json.dump({"n_projected_vectors": len(projected_ids),
          "autoencoder": {"roc_auc": roc, "pr_auc": pr, "recall_at_0.5fpr": rec,
                         "precision": prec, "threshold": det.threshold}},
         open("artifacts/defend_ae_eval.json", "w"), indent=1)
print("-> artifacts/defend_ae_eval.json, artifacts/model/anomaly_ae.pt", flush=True)
