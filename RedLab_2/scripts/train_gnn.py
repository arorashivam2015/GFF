"""Train the GNN and the tabular graph-proxy baseline on identical splits,
report the comparison that is this solution's headline result."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from redlab.defend.gnn import GraphDetector
from redlab.defend.graph_features import build_graph_proxy_features, feature_names

t0 = time.time()
df = pd.read_parquet("data/processed/graph_combined.parquet")
cut = df.timestamp.quantile(0.7)
train_df, test_df = df[df.timestamp <= cut], df[df.timestamp > cut]
print(f"[{time.time()-t0:.0f}s] {len(df):,} edges | train {len(train_df):,} "
     f"| test {len(test_df):,} ({int(test_df.is_fraud.sum())} fraud)")


def recall_precision_at_fpr(y, p, fpr):
    neg = p[y == 0]
    thresh = np.quantile(neg, 1 - fpr) if len(neg) else np.inf
    flagged = p >= thresh
    tp = int((flagged & (y == 1)).sum())
    return tp / max(int((y == 1).sum()), 1), tp / max(int(flagged.sum()), 1)


# --- GNN -------------------------------------------------------------------
print("\ntraining GNN...")
gd = GraphDetector(epochs=15, seed=0).fit(df, train_df)
p_gnn = gd.score(test_df)
y = test_df.is_fraud.to_numpy()
print(f"[{time.time()-t0:.0f}s] GNN trained and scored")

cold = gd.cold_start_mask(test_df)
print(f"cold-start test edges (neither endpoint seen in train): "
     f"{cold.sum():,} / {len(test_df):,} ({100*cold.mean():.1f}%), "
     f"of which fraud: {int(y[cold].sum())}/{int(y.sum())}")

roc_g = roc_auc_score(y, p_gnn)
pr_g = average_precision_score(y, p_gnn)
rec_g, prec_g = recall_precision_at_fpr(y, p_gnn, 0.005)
print(f"GNN (all test edges):        ROC-AUC {roc_g:.4f}  PR-AUC {pr_g:.4f}  "
     f"recall@0.5%FPR {rec_g*100:.1f}%  precision {prec_g*100:.1f}%")
if (~cold).sum() and y[~cold].sum():
    roc_w, pr_w = roc_auc_score(y[~cold], p_gnn[~cold]), average_precision_score(y[~cold], p_gnn[~cold])
    print(f"GNN (warm edges only):        ROC-AUC {roc_w:.4f}  PR-AUC {pr_w:.4f}")
if cold.sum() and y[cold].sum():
    roc_c, pr_c = roc_auc_score(y[cold], p_gnn[cold]), average_precision_score(y[cold], p_gnn[cold])
    print(f"GNN (cold-start edges only):  ROC-AUC {roc_c:.4f}  PR-AUC {pr_c:.4f}")

# --- tabular graph-proxy baseline -------------------------------------------
print("\ntraining tabular graph-proxy baseline...")
import lightgbm as lgb
F = build_graph_proxy_features(df)
tr, te = F[F.timestamp <= cut], F[F.timestamp > cut]
cols = feature_names(F)
Xtr, Xte = tr[cols].copy(), te[cols].copy()
for c in ["src_type", "dst_type"]:
    Xtr[c] = Xtr[c].astype("category")
    Xte[c] = Xte[c].astype("category")
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                         random_state=0, verbosity=-1).fit(Xtr, tr.is_fraud)
p_proxy = clf.predict_proba(Xte)[:, 1]
roc_p = roc_auc_score(te.is_fraud, p_proxy)
pr_p = average_precision_score(te.is_fraud, p_proxy)
rec_p, prec_p = recall_precision_at_fpr(te.is_fraud.to_numpy(), p_proxy, 0.005)
print(f"[{time.time()-t0:.0f}s] tabular proxy:  ROC-AUC {roc_p:.4f}  PR-AUC {pr_p:.4f}  "
     f"recall@0.5%FPR {rec_p*100:.1f}%  precision {prec_p*100:.1f}%")

print("\n" + "="*74)
print("HEADLINE: GNN vs. tabular graph-proxy, identical split")
print("="*74)
print(f"  {'metric':<20}{'GNN':>12}{'tabular proxy':>16}{'gap':>10}")
print(f"  {'ROC-AUC':<20}{roc_g:>12.4f}{roc_p:>16.4f}{roc_g-roc_p:>+10.4f}")
print(f"  {'PR-AUC':<20}{pr_g:>12.4f}{pr_p:>16.4f}{pr_g-pr_p:>+10.4f}")
print(f"  {'recall@0.5%FPR':<20}{rec_g*100:>11.1f}%{rec_p*100:>15.1f}%{(rec_g-rec_p)*100:>+9.1f}pp")

import json
pathlib.Path("artifacts").mkdir(exist_ok=True)
json.dump({
    "gnn": {"roc_auc": roc_g, "pr_auc": pr_g, "recall_at_0.5fpr": rec_g, "precision": prec_g,
           "cold_start_edges": int(cold.sum()), "cold_start_fraud": int(y[cold].sum())},
    "tabular_proxy": {"roc_auc": roc_p, "pr_auc": pr_p, "recall_at_0.5fpr": rec_p, "precision": prec_p},
}, open("artifacts/gnn_vs_proxy.json", "w"), indent=1)
print(f"\n-> artifacts/gnn_vs_proxy.json  ({time.time()-t0:.0f}s total)")
