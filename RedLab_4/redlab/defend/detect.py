"""Detection model and its evaluation.

The model here is deliberately ordinary - gradient-boosted trees on causal
features, which is what payment risk teams actually run. The work that matters
is in how it is evaluated.

WHY ROC-AUC IS NOT THE HEADLINE
-------------------------------
Measured on this corpus, a model using only raw transaction fields reaches
ROC-AUC 0.966. The reference corpus's own labelled fraud scores 0.966 on the
identical feature set. At sub-1% prevalence, ROC-AUC is dominated by the
enormous negative class and flatters everything. The metrics that discriminate
between a useful model and a useless one are:

  PR-AUC              sensitive to prevalence, unlike ROC-AUC
  recall @ fixed FPR  how payments risk teams actually specify a model, because
                      false-positive budget is the binding constraint
  alert rate          what the review queue actually receives

WHY LEAVE-ONE-FAMILY-OUT IS THE REAL TEST
-----------------------------------------
The challenge asks for detection of EMERGING fraud. A detector trained on the
same attack families it is scored against is answering an easier question than
the one posed. `leave_one_family_out` removes entire attack families from
training and scores only on those held-out families, so the reported number
describes generalisation to attacks the model has genuinely never seen. It is
substantially worse than the in-distribution number. That gap is the finding,
not an embarrassment to be hidden.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import average_precision_score, roc_auc_score

FPR_TARGETS = (0.001, 0.005, 0.01)
META_COLS = {"txn_id", "timestamp", "user_id", "merchant_id", "device_id",
             "is_fraud", "attack_id"}


class EvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    n_test: int
    n_positive: int
    prevalence: float
    roc_auc: float
    pr_auc: float
    recall_at_fpr: Dict[str, float]
    precision_at_fpr: Dict[str, float]
    per_vector_recall: Dict[str, float] = {}

    def render(self) -> str:
        out = [f"\n{self.label}",
               f"  n={self.n_test:,}  positives={self.n_positive:,}  "
               f"prevalence={self.prevalence*100:.3f}%",
               f"  ROC-AUC {self.roc_auc:.4f}    PR-AUC {self.pr_auc:.4f}"]
        for k in self.recall_at_fpr:
            out.append(f"  @FPR {k:>6s}:  recall {self.recall_at_fpr[k]*100:5.1f}%   "
                       f"precision {self.precision_at_fpr[k]*100:5.1f}%")
        return "\n".join(out)


def recall_precision_at_fpr(y: np.ndarray, p: np.ndarray, fpr: float):
    """Operating point defined by false-positive budget, not by score cutoff."""
    neg = p[y == 0]
    if len(neg) == 0:
        return 0.0, 0.0
    thresh = np.quantile(neg, 1.0 - fpr)
    flagged = p >= thresh
    tp = int((flagged & (y == 1)).sum())
    recall = tp / max(int((y == 1).sum()), 1)
    precision = tp / max(int(flagged.sum()), 1)
    return float(recall), float(precision)


@dataclass
class Detector:
    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 63
    seed: int = 0
    model: object = field(default=None, init=False)
    features: List[str] = field(default_factory=list, init=False)

    def _prep(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.features].copy()
        for c in X.columns:
            if str(X[c].dtype) in ("object", "category"):
                X[c] = X[c].astype("category")
        return X

    def fit(self, train: pd.DataFrame) -> "Detector":
        import lightgbm as lgb

        self.features = [c for c in train.columns if c not in META_COLS]
        X = self._prep(train)
        y = train["is_fraud"].to_numpy()
        # scale_pos_weight left at 1: with PR-AUC and FPR-anchored operating
        # points, rebalancing changes calibration without improving ranking.
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=50,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
            random_state=self.seed, verbosity=-1).fit(X, y)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prep(frame))[:, 1]

    def evaluate(self, test: pd.DataFrame, label: str,
                 per_vector: bool = False) -> EvalResult:
        p = self.score(test)
        y = test["is_fraud"].to_numpy()

        rec, prec = {}, {}
        for f in FPR_TARGETS:
            r, pr = recall_precision_at_fpr(y, p, f)
            rec[f"{f*100:.1f}%"] = r
            prec[f"{f*100:.1f}%"] = pr

        pv: Dict[str, float] = {}
        if per_vector and "attack_id" in test:
            neg = p[y == 0]
            thresh = np.quantile(neg, 1.0 - 0.005) if len(neg) else np.inf
            t = test.assign(_p=p)
            for vid, g in t[t.is_fraud == 1].groupby("attack_id"):
                pv[str(vid)] = float((g._p >= thresh).mean())

        return EvalResult(
            label=label, n_test=len(test), n_positive=int(y.sum()),
            prevalence=float(y.mean()),
            roc_auc=float(roc_auc_score(y, p)) if y.sum() else float("nan"),
            pr_auc=float(average_precision_score(y, p)) if y.sum() else float("nan"),
            recall_at_fpr=rec, precision_at_fpr=prec, per_vector_recall=pv)

    def importances(self, top: int = 15) -> pd.Series:
        g = self.model.booster_.feature_importance("gain")
        return pd.Series(g, index=self.features).sort_values(ascending=False).head(top)


def temporal_split(frame: pd.DataFrame, test_frac: float = 0.3):
    """Split by time. Random splits leak: campaigns span many rows, so a random
    split puts earlier events of the same campaign in train and later ones in
    test, which is not a situation a deployed model ever faces."""
    cut = frame["timestamp"].quantile(1 - test_frac)
    return frame[frame.timestamp <= cut], frame[frame.timestamp > cut]


def mechanism_holdout(frame: pd.DataFrame, taxonomy, axis: str,
                      values: Sequence[str], test_frac: float = 0.3):
    """Hold out a GENERATIVE AXIS rather than a family label.

    Measured on this corpus, holding out family labels costs the detector only
    ~0.02 PR-AUC, because all 42 vectors are rendered by one generic engine -
    a family label bundles parameter combinations, not distinct mechanisms.
    Holding out amount profiles instead costs ~0.20 PR-AUC. The harder number
    is the one worth reporting for "detects emerging fraud", because an
    unfamiliar attack differs in mechanism, not in what we chose to call it.

    axis: "amount_profile" or "temporal_shape".
    """
    prof = {v.id: getattr(v.simulation, axis).value for v in taxonomy}
    tag = frame["attack_id"].map(lambda a: prof.get(a, "") if pd.notna(a) else "")
    hold = tag.isin(values)

    cut = frame["timestamp"].quantile(1 - test_frac)
    train = frame[(frame.timestamp <= cut) & ~(frame.is_fraud.eq(1) & hold)]
    test = frame[(frame.timestamp > cut) & (frame.is_fraud.eq(0) | hold)]
    return train, test


def leave_one_family_out(frame: pd.DataFrame, holdout_prefixes: Sequence[str],
                         test_frac: float = 0.3, seed: int = 0):
    """Train without entire attack families; score only on those families.

    Legitimate traffic is split temporally as usual so the false-positive
    denominator stays realistic. Held-out-family fraud is removed from train
    entirely - not merely down-weighted - so the model has no exposure to it.
    """
    is_holdout = frame["attack_id"].fillna("").str.split("-").str[1].isin(holdout_prefixes)
    tr, te = temporal_split(frame, test_frac)

    train = tr[~(tr.is_fraud.eq(1) & is_holdout.reindex(tr.index, fill_value=False))]
    test = te[(te.is_fraud.eq(0)) |
              (te.is_fraud.eq(1) & is_holdout.reindex(te.index, fill_value=False))]
    return train, test
