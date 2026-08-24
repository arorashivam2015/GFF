"""Headline comparison: local-only vs. federated vs. centralized-oracle, on
the cross-institution attack subset RedLab_1's own taxonomy already flags as
exploiting single-institution blind spots."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from redlab.defend.features import build_features, feature_names
from redlab.defend.federated import federated_average, fit_centralized_oracle, fit_local_models
from redlab.sim.institutions import assign_institutions, cross_institution_spread

N_INSTITUTIONS = 4
CROSS_INST_VECTORS = ["PF-IND-005", "PF-CT-001"]  # mule layering, distributed BIN testing


def recall_precision_at_fpr(y, p, fpr):
    neg = p[y == 0]
    thresh = np.quantile(neg, 1 - fpr) if len(neg) else np.inf
    flagged = p >= thresh
    tp = int((flagged & (y == 1)).sum())
    return tp / max(int((y == 1).sum()), 1), tp / max(int(flagged.sum()), 1)


df = pd.read_parquet("data/processed/world_attacked.parquet")
df = assign_institutions(df, N_INSTITUTIONS)
print(f"{len(df):,} rows partitioned across {N_INSTITUTIONS} synthetic institutions")

print("\ncross-institution spread of the two hero vectors (per taxonomy's own hypotheses):")
for vid in CROSS_INST_VECTORS:
    spread = cross_institution_spread(df, vid)
    print(f"  {vid}: {spread['n_events']} events span "
         f"{spread['n_issuers']} issuers, {spread['n_acquirers']} acquirers "
         f"(of {N_INSTITUTIONS} total)")

feats = build_features(df)
feats["issuer"] = df["issuer"].to_numpy()
feats["acquirer"] = df["acquirer"].to_numpy()
cols = feature_names(feats)
cols = [c for c in cols if c not in ("mcc", "channel")]  # keep the linear model numeric-only

# The two hero vectors are short, early-window campaigns (burst tempo, per
# their own taxonomy spec) that land entirely before a 70th-percentile
# temporal cutoff - a straight temporal split leaves zero of their fraud in
# the test set, which tests nothing. This solution's question is "does
# institutional silo-ing hurt detection of cross-institution attacks," not
# "does the model generalise across time" (that's RedLab_1's and RedLab_3's
# question) - so these two vectors are split 70/30 on their OWN events,
# stratified, guaranteeing both sides see real examples; everything else
# still uses the temporal split for the general training population.
rng = np.random.default_rng(0)
is_hero = feats.attack_id.isin(CROSS_INST_VECTORS)
hero = feats[is_hero]
hero_test_idx = rng.choice(hero.index, size=int(len(hero) * 0.3), replace=False)
hero_train = hero.drop(hero_test_idx)
hero_test = hero.loc[hero_test_idx]

cut = feats[~is_hero].timestamp.quantile(0.7)
base_train = feats[~is_hero & (feats.timestamp <= cut)]
base_test = feats[~is_hero & (feats.timestamp > cut)]

train = pd.concat([base_train, hero_train], ignore_index=True)
cross_test = pd.concat([base_test[base_test.is_fraud == 0], hero_test], ignore_index=True)
y_cross = cross_test.is_fraud.to_numpy()
print(f"\ntrain: {len(train):,} rows | cross-institution test: {len(cross_test):,} rows "
     f"({int(y_cross.sum())} fraud from {CROSS_INST_VECTORS})")

# --- local-only: each institution scores only with its OWN model -----------
locals_ = fit_local_models(train, cols, "issuer", seed=0)
print(f"\nfit {len(locals_)} local (per-issuer) models")
local_scores = np.zeros(len(cross_test))
for inst, model in locals_.items():
    mask = cross_test["issuer"].to_numpy() == inst
    if mask.any():
        local_scores[mask] = model.score(cross_test[mask])
roc_local = roc_auc_score(y_cross, local_scores)
pr_local = average_precision_score(y_cross, local_scores)
rec_local, prec_local = recall_precision_at_fpr(y_cross, local_scores, 0.01)

# --- federated: one round of coefficient averaging across institutions -----
n_by_inst = train.groupby("issuer").size().to_dict()
fed_model = federated_average(locals_, weight_by_n=n_by_inst)
fed_scores = fed_model.score(cross_test)
roc_fed = roc_auc_score(y_cross, fed_scores)
pr_fed = average_precision_score(y_cross, fed_scores)
rec_fed, prec_fed = recall_precision_at_fpr(y_cross, fed_scores, 0.01)

# --- centralized oracle: full data pooling, upper-bound reference only -----
oracle = fit_centralized_oracle(train, cols, seed=0)
oracle_scores = oracle.score(cross_test)
roc_oracle = roc_auc_score(y_cross, oracle_scores)
pr_oracle = average_precision_score(y_cross, oracle_scores)
rec_oracle, prec_oracle = recall_precision_at_fpr(y_cross, oracle_scores, 0.01)

print(f"\n{'='*72}")
print(f"HEADLINE: recall @ 1% FPR on cross-institution fraud "
     f"({' + '.join(CROSS_INST_VECTORS)})")
print(f"{'='*72}")
print(f"  {'model':<22}{'ROC-AUC':>10}{'PR-AUC':>10}{'recall@1%FPR':>15}")
print(f"  {'local-only (status quo)':<22}{roc_local:>10.4f}{pr_local:>10.4f}{rec_local*100:>14.1f}%")
print(f"  {'federated':<22}{roc_fed:>10.4f}{pr_fed:>10.4f}{rec_fed*100:>14.1f}%")
print(f"  {'centralized oracle':<22}{roc_oracle:>10.4f}{pr_oracle:>10.4f}{rec_oracle*100:>14.1f}%")

recovery = ((rec_fed - rec_local) / max(rec_oracle - rec_local, 1e-9)) * 100
print(f"\nfederation recovers {recovery:.0f}% of the local->oracle recall gap")

pathlib.Path("artifacts").mkdir(exist_ok=True)
json.dump({
    "n_institutions": N_INSTITUTIONS,
    "cross_institution_vectors": CROSS_INST_VECTORS,
    "spread": {vid: cross_institution_spread(df, vid) for vid in CROSS_INST_VECTORS},
    "local": {"roc_auc": roc_local, "pr_auc": pr_local, "recall_at_1fpr": rec_local,
             "precision": prec_local},
    "federated": {"roc_auc": roc_fed, "pr_auc": pr_fed, "recall_at_1fpr": rec_fed,
                 "precision": prec_fed},
    "centralized_oracle": {"roc_auc": roc_oracle, "pr_auc": pr_oracle,
                          "recall_at_1fpr": rec_oracle, "precision": prec_oracle},
    "recovery_pct": recovery,
}, open("artifacts/eval_results.json", "w"), indent=1)
print("\n-> artifacts/eval_results.json")
