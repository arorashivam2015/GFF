"""Lean guardrails on the one thing this solution's headline number depends
on: that projected-maturity fraud genuinely never touches training."""
import pathlib
import numpy as np
import pandas as pd
import pytest

from redlab.defend.anomaly import AnomalyEnsemble
from redlab.defend.features import feature_names
from redlab.defend.zero_exposure import verify_no_leakage, zero_exposure_split
from redlab.taxonomy.loader import Taxonomy

pytestmark = pytest.mark.skipif(
    not pathlib.Path("data/processed/features.parquet").exists(),
    reason="features not built; run scripts/evaluate.py's upstream steps first",
)


@pytest.fixture(scope="module")
def data():
    return pd.read_parquet("data/processed/features.parquet"), Taxonomy.load()


def test_split_has_projected_vectors(data):
    F, tax = data
    assert sum(1 for v in tax if v.maturity.value == "projected") > 0


def test_no_projected_fraud_in_train(data):
    F, tax = data
    train, test = zero_exposure_split(F, tax)
    verify_no_leakage(train, tax)  # raises on any leak


def test_leakage_check_actually_detects_a_real_leak(data):
    """The verifier must fail on a deliberately corrupted split, or it isn't
    testing anything."""
    F, tax = data
    train, _ = zero_exposure_split(F, tax)
    projected_id = next(v.id for v in tax if v.maturity.value == "projected")
    corrupted = pd.concat([train, F[F.attack_id == projected_id].head(1)])
    with pytest.raises(AssertionError):
        verify_no_leakage(corrupted, tax)


def test_test_set_is_only_projected_or_legit(data):
    F, tax = data
    _, test = zero_exposure_split(F, tax)
    projected_ids = {v.id for v in tax if v.maturity.value == "projected"}
    fraud_ids = set(test[test.is_fraud == 1].attack_id.unique())
    assert fraud_ids <= projected_ids


def test_anomaly_ensemble_fits_on_legit_only_and_scores(data):
    F, tax = data
    train, test = zero_exposure_split(F, tax)
    legit = train[train.is_fraud == 0].sample(min(20000, (train.is_fraud == 0).sum()),
                                               random_state=0)
    ens = AnomalyEnsemble(seed=0).fit(legit, feature_names(F))
    scores = ens.score(test.head(500))
    assert len(scores) == 500
    assert np.isfinite(scores).all()


def test_conformal_coverage_is_close_to_target(data):
    """The distribution-free guarantee should roughly hold - not exactly
    (finite-sample noise), but within a wide, honest tolerance."""
    F, tax = data
    train, _ = zero_exposure_split(F, tax)
    legit = train[train.is_fraud == 0]
    rng = np.random.default_rng(1)
    holdout_idx = rng.choice(len(legit), size=min(5000, len(legit) // 4), replace=False)
    fit_part = legit.drop(legit.index[holdout_idx])
    holdout_part = legit.iloc[holdout_idx]
    ens = AnomalyEnsemble(target_fpr=0.02, seed=1).fit(fit_part, feature_names(F))
    coverage = ens.verify_coverage(holdout_part)
    assert 0.005 < coverage < 0.06, f"empirical FPR {coverage} far from target 0.02"
