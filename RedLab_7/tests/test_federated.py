"""Lean guardrails on the federation mechanics and institution partitioning."""
import pathlib
import numpy as np
import pandas as pd
import pytest

from redlab.defend.federated import federated_average, fit_centralized_oracle, fit_local_models
from redlab.sim.institutions import assign_institutions, cross_institution_spread

pytestmark = pytest.mark.skipif(
    not pathlib.Path("data/processed/world_attacked.parquet").exists(),
    reason="world not built",
)


@pytest.fixture(scope="module")
def df():
    d = pd.read_parquet("data/processed/world_attacked.parquet")
    return assign_institutions(d, n_institutions=4)


def test_institution_assignment_is_deterministic(df):
    again = assign_institutions(df.drop(columns=["issuer", "acquirer"]), n_institutions=4)
    assert (again.issuer == df.issuer).all()
    assert (again.acquirer == df.acquirer).all()


def test_institution_assignment_spreads_population(df):
    """A degenerate hash that puts everyone in one bucket would make the
    entire premise of this solution vacuous."""
    assert df.issuer.nunique() == 4
    assert df.acquirer.nunique() == 4
    counts = df.issuer.value_counts(normalize=True)
    assert counts.min() > 0.15  # roughly even, not a pathological skew


def test_hero_vectors_span_multiple_institutions(df):
    for vid in ["PF-IND-005", "PF-CT-001"]:
        spread = cross_institution_spread(df, vid)
        assert spread["n_issuers"] > 1
        assert spread["n_acquirers"] > 1


def test_federated_average_is_a_real_weighted_mean():
    """federated_average must actually average, not just return one local
    model - checked against a hand-computed weighted mean on synthetic
    coefficient vectors."""
    from sklearn.preprocessing import StandardScaler
    from redlab.defend.federated import LinearModel

    def fake(coef, intercept):
        s = StandardScaler()
        s.mean_, s.scale_ = np.zeros(len(coef)), np.ones(len(coef))
        return LinearModel(scaler=s, coef=np.array(coef), intercept=intercept,
                          feature_cols=["a", "b"])

    locals_ = {0: fake([1.0, 0.0], 0.0), 1: fake([0.0, 1.0], 2.0)}
    fed = federated_average(locals_, weight_by_n={0: 1, 1: 1})
    assert np.allclose(fed.coef, [0.5, 0.5])
    assert fed.intercept == pytest.approx(1.0)

    fed_weighted = federated_average(locals_, weight_by_n={0: 3, 1: 1})
    assert fed_weighted.coef[0] > fed.coef[0]  # heavier institution pulls the average toward it


def test_no_raw_data_crosses_the_federation_boundary():
    """federated_average's signature only accepts already-fit LinearModels -
    structurally, it cannot see a training DataFrame, which is the actual
    privacy property this solution claims."""
    import inspect
    sig = inspect.signature(federated_average)
    for pname, p in sig.parameters.items():
        assert "DataFrame" not in str(p.annotation), \
            f"federated_average accepts raw data via '{pname}' - breaks the privacy claim"
