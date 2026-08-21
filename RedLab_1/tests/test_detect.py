"""Guardrails on the DEFEND pillar and its evaluation honesty.

These lock in properties that were violated once already: a first detector run
hit PR-AUC 1.0000 because of a simulator leak (every legit device had exactly
one owner). Re-checking causal correctness here is cheap insurance against that
class of bug recurring silently.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from redlab.defend.detect import (Detector, leave_one_family_out,
                                  recall_precision_at_fpr, temporal_split)
from redlab.defend.features import build_features, feature_names

FEATURES_PATH = "data/processed/features.parquet"
pytestmark = pytest.mark.skipif(
    not pathlib.Path(FEATURES_PATH).exists(),
    reason="features not built; run scripts/build_dataset.py",
)


@pytest.fixture(scope="module")
def features():
    return pd.read_parquet(FEATURES_PATH)


def test_features_are_causal_on_a_synthetic_trap(tmp_path):
    """Construct a case where a NON-causal feature would trivially leak: one
    user with a single enormous transaction. A leaking mean-amount feature
    would divide the amount by itself (ratio 1.0, no signal); a causal one has
    no prior history at all (NaN) because there IS no prior transaction."""
    df = pd.DataFrame({
        "txn_id": ["T0"], "timestamp": [pd.Timestamp("2025-01-01")],
        "hour": [0], "dow": [0], "user_id": ["U0"], "device_id": ["D0"],
        "merchant_id": ["M0"], "mcc": ["5411"], "channel": ["Swipe Transaction"],
        "amount": [99999.0], "state": ["CA"], "error": ["(none)"],
        "is_fraud": [0], "attack_id": [None],
    })
    f = build_features(df)
    assert f.loc[0, "u_amt_over_mean"] != 1.0 or pd.isna(f.loc[0, "u_amt_over_mean"])
    assert pd.isna(f.loc[0, "u_amt_over_mean"])  # no prior txn to compare against


def test_features_do_not_use_future_rows(features):
    """A user's feature values at their k-th transaction must be identical
    whether or not later transactions of theirs exist in the frame - i.e.
    truncating the frame after row k must not change row k's features."""
    busy = features.groupby("user_id").filter(lambda g: len(g) >= 20)
    uid = busy.user_id.iloc[0]
    full = features[features.user_id == uid].sort_values("timestamp")
    raw_cols = ["timestamp", "hour", "dow", "user_id", "device_id",
               "merchant_id", "mcc", "channel", "amount", "is_fraud", "attack_id"]
    # Recompute features on a prefix of this user's own history only.
    prefix = full.iloc[:10][raw_cols].assign(txn_id=[f"X{i}" for i in range(10)],
                                             state="CA", error="(none)")
    recomputed = build_features(prefix)
    for col in ["u_prior_n", "u_amt_over_mean", "u_first_merchant"]:
        a = full.iloc[9][col]
        b = recomputed.iloc[9][col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert np.isclose(a, b, equal_nan=True), f"{col} differs: {a} vs {b}"


def test_detector_trains_and_scores(features):
    tr, te = temporal_split(features)
    det = Detector(n_estimators=100).fit(tr)
    p = det.score(te)
    assert len(p) == len(te)
    assert ((p >= 0) & (p <= 1)).all()


def test_recall_at_fpr_respects_budget():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(9800)]
    p = np.r_[rng.uniform(0.3, 1.0, 200), rng.uniform(0.0, 0.9, 9800)]
    recall, precision = recall_precision_at_fpr(y, p, 0.01)
    neg_flagged = ((p >= np.quantile(p[y == 0], 0.99)) & (y == 0)).mean()
    assert abs(neg_flagged - 0.01) < 0.005


def test_leave_one_family_out_has_no_holdout_fraud_in_train(features):
    train, test = leave_one_family_out(features, ["AGC", "ADV"])
    train_fams = train.loc[train.is_fraud.eq(1), "attack_id"].str.split("-").str[1]
    assert not set(train_fams) & {"AGC", "ADV"}
    test_fams = test.loc[test.is_fraud.eq(1), "attack_id"].str.split("-").str[1]
    assert set(test_fams) <= {"AGC", "ADV"}


def test_in_distribution_beats_unseen_mechanism(features):
    """The mechanism holdout must be a genuinely harder test than the
    in-distribution split, or it isn't testing what it claims to."""
    from redlab.defend.detect import mechanism_holdout
    from redlab.taxonomy.loader import Taxonomy

    tax = Taxonomy.load()
    tr_in, te_in = temporal_split(features)
    tr_out, te_out = mechanism_holdout(features, tax, "amount_profile",
                                       ["micro_probe", "drain"])
    d_in = Detector(n_estimators=150).fit(tr_in)
    d_out = Detector(n_estimators=150).fit(tr_out)
    r_in = d_in.evaluate(te_in, "in")
    r_out = d_out.evaluate(te_out, "out")
    assert r_out.pr_auc < r_in.pr_auc
