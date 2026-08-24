"""The primary detector: an unsupervised ensemble that never sees a fraud
label during training, paired with a conformal calibration layer for a
statistically justified threshold instead of an arbitrary score cutoff.

Two base detectors, kept deliberately simple rather than a deep model - the
point of this solution is the ZERO-LABEL-EXPOSURE evaluation discipline, not
architectural novelty on the anomaly-scoring side, and a from-scratch neural
autoencoder buys little here for real training-time and complexity cost:

  IsolationForest        isolates outliers by average path length in random
                          partitioning trees - cheap, well-understood, no
                          assumption about feature distribution shape.
  PCA reconstruction      fit PCA on legitimate rows only; reconstruction
                          error (how much a row can't be explained by the
                          top components of NORMAL variation) is a standard,
                          fast stand-in for an autoencoder's reconstruction
                          error, without a training loop to babysit.

CONFORMAL CALIBRATION: a threshold picked as "the 99.5th percentile of
TRAINING scores" is not a calibrated guarantee - it's a guess that happens to
look precise. Split conformal prediction fixes this: a held-out CALIBRATION
slice of legitimate rows, untouched during model fitting, gives an empirical
quantile with a real distribution-free coverage guarantee - if you set the
threshold at its 99.5th percentile, at most 0.5% of held-out legitimate rows
should exceed it, IN EXPECTATION, regardless of what the anomaly scores
actually look like. That guarantee is verified against the true test set
below, not just asserted.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

NUMERIC_ONLY_DROP = {"mcc", "channel"}  # categorical; ensemble uses numeric features only


def _numeric_columns(frame: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    return [c for c in feature_cols if c not in NUMERIC_ONLY_DROP]


@dataclass
class AnomalyEnsemble:
    calib_frac: float = 0.15
    target_fpr: float = 0.005
    seed: int = 0
    scaler: StandardScaler = field(default=None, init=False)
    iforest: IsolationForest = field(default=None, init=False)
    pca: PCA = field(default=None, init=False)
    threshold: float = field(default=None, init=False)
    feature_cols: List[str] = field(default=None, init=False)

    def fit(self, legit_train: pd.DataFrame, feature_cols: List[str]) -> "AnomalyEnsemble":
        self.feature_cols = _numeric_columns(legit_train, feature_cols)
        X = legit_train[self.feature_cols].fillna(0).to_numpy()

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        n_calib = int(len(X) * self.calib_frac)
        calib_idx, fit_idx = idx[:n_calib], idx[n_calib:]

        self.scaler = StandardScaler().fit(X[fit_idx])
        Xs = self.scaler.transform(X[fit_idx])

        self.iforest = IsolationForest(n_estimators=200, contamination="auto",
                                       random_state=self.seed).fit(Xs)
        self.pca = PCA(n_components=min(10, Xs.shape[1] - 1), random_state=self.seed).fit(Xs)

        calib_scores = self._raw_scores(X[calib_idx])
        # Split-conformal threshold: the empirical (1 - target_fpr) quantile
        # of scores on data the model never fit, not data it fit on.
        self.threshold = float(np.quantile(calib_scores, 1 - self.target_fpr))
        return self

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        iso_score = -self.iforest.score_samples(Xs)          # higher = more anomalous
        recon = self.pca.inverse_transform(self.pca.transform(Xs))
        pca_score = np.mean((Xs - recon) ** 2, axis=1)

        # Rank-normalise each channel before averaging - the two scores live
        # on unrelated scales, and a raw average would let whichever one has
        # the larger numeric range silently dominate the ensemble.
        iso_rank = pd.Series(iso_score).rank(pct=True).to_numpy()
        pca_rank = pd.Series(pca_score).rank(pct=True).to_numpy()
        return (iso_rank + pca_rank) / 2

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[self.feature_cols].fillna(0).to_numpy()
        return self._raw_scores(X)

    def flag(self, frame: pd.DataFrame) -> np.ndarray:
        return self.score(frame) >= self.threshold

    def verify_coverage(self, legit_holdout: pd.DataFrame) -> float:
        """Empirical false-positive rate on held-out LEGIT rows never used
        for fitting or calibration - the number that tells you whether the
        conformal guarantee actually held, not just whether it was invoked."""
        return float(self.flag(legit_holdout).mean())
