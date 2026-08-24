"""Three models on the identical cross-institution test set: local-only (the
realistic status quo), federated (privacy-preserving), and a centralized
oracle (unrealistic, included as an upper-bound reference only).

A linear model (logistic regression), not a GBM, is used deliberately: model
AVERAGING across institutions - the actual mechanic of FedAvg - is a
well-defined operation on a coefficient vector. It has no equivalent
definition for an ensemble of trees, where "averaging two GBMs" is not a
standard, well-defined operation the way averaging two weight vectors is.
Using a simpler model class here is what makes the federation itself
genuine, rather than an approximation dressed up as one.

Single communication round: each institution trains locally to convergence,
then coefficients are averaged once. This is FedAvg's simplest case (one
round, full local convergence per round) rather than the many-round protocol
production federated systems use - stated plainly as a simplification, not
hidden as if it were the full protocol.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class LinearModel:
    """A thin wrapper so local, federated, and centralized models share one
    scoring interface regardless of how their coefficients were produced."""
    scaler: StandardScaler
    coef: np.ndarray
    intercept: float
    feature_cols: List[str]

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(frame[self.feature_cols].fillna(0).to_numpy())
        z = X @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


def _fit_local(frame: pd.DataFrame, feature_cols: List[str], seed: int) -> LinearModel:
    X = frame[feature_cols].fillna(0).to_numpy()
    y = frame["is_fraud"].to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed).fit(Xs, y)
    return LinearModel(scaler=scaler, coef=clf.coef_[0], intercept=float(clf.intercept_[0]),
                       feature_cols=feature_cols)


def fit_local_models(train: pd.DataFrame, feature_cols: List[str], institution_col: str,
                     seed: int = 0) -> Dict[int, LinearModel]:
    """One model per institution, trained ONLY on that institution's own data
    - the realistic status quo before any collaboration."""
    return {inst: _fit_local(g, feature_cols, seed)
           for inst, g in train.groupby(institution_col) if g["is_fraud"].sum() > 5}


def federated_average(locals_: Dict[int, LinearModel], weight_by_n: Dict[int, int] = None
                      ) -> LinearModel:
    """FedAvg: average coefficient vectors across institutions, weighted by
    each institution's local sample count if given (larger institutions
    contribute proportionally more, matching the real FedAvg weighting rule).
    Only coefficients and the intercept cross the boundary - never raw data."""
    models = list(locals_.values())
    if weight_by_n:
        w = np.array([weight_by_n[k] for k in locals_], dtype=float)
        w = w / w.sum()
    else:
        w = np.full(len(models), 1.0 / len(models))
    coef = np.average([m.coef for m in models], axis=0, weights=w)
    intercept = float(np.average([m.intercept for m in models], weights=w))
    # Scaler: average feature means/scales across institutions, since a
    # federated deployment scores with one shared preprocessing step, not
    # each institution's own local scaler.
    means = np.average([m.scaler.mean_ for m in models], axis=0, weights=w)
    scales = np.average([m.scaler.scale_ for m in models], axis=0, weights=w)
    shared_scaler = StandardScaler()
    shared_scaler.mean_, shared_scaler.scale_ = means, scales
    shared_scaler.var_ = scales ** 2
    shared_scaler.n_features_in_ = len(means)
    return LinearModel(scaler=shared_scaler, coef=coef, intercept=intercept,
                       feature_cols=models[0].feature_cols)


def fit_centralized_oracle(train: pd.DataFrame, feature_cols: List[str], seed: int = 0
                           ) -> LinearModel:
    """Full data pooling across institutions - not realistic (this is exactly
    the data-sharing constraint federation exists to avoid) but included as
    an upper-bound reference so the federated result can be judged against
    what it recovers, not read in isolation."""
    return _fit_local(train, feature_cols, seed)
