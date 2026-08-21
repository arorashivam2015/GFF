"""Structural guardrails on the GENERATE pillar.

These assert the joint dependencies that fidelity work established. Each one
corresponds to a defect that was measured and fixed, so a regression here means
the discriminator AUC is about to get worse.
"""

import numpy as np
import pytest

from redlab.sim.conditionals import ConditionalProfile
from redlab.sim.world import PaymentWorld, WorldConfig

PROFILE = "data/processed/conditional_profile.json"
pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(PROFILE).exists(),
    reason="conditional profile not built; run python -m redlab.sim.conditionals",
)


@pytest.fixture(scope="module")
def world():
    cfg = WorldConfig(n_cardholders=1500, n_merchants=3000, days=90, seed=7)
    return PaymentWorld(ConditionalProfile.load(PROFILE), cfg)


@pytest.fixture(scope="module")
def txns(world):
    return world.generate()


def test_generates_transactions(txns):
    assert len(txns) > 10_000
    assert txns.amount.notna().all()


def test_entities_are_persistent(txns):
    """Users must recur. A generator emitting one txn per user has no
    behavioural baseline for a detector to learn."""
    per_user = txns.groupby("user_id").size()
    assert per_user.median() > 20


def test_merchant_is_mcc_exclusive(txns):
    """Reference corpus: 99.3% of merchants trade under exactly one MCC."""
    per_merchant_mccs = txns.groupby("merchant_id")["mcc"].nunique()
    assert (per_merchant_mccs == 1).mean() > 0.99


def test_amount_depends_on_category(txns):
    """The dependency a marginals-only generator lacks. Category medians must
    differ materially, not be a single pooled distribution."""
    med = txns[txns.amount > 0].groupby("mcc").amount.median()
    assert med.max() / med.min() > 3.0


def test_hour_depends_on_category(txns):
    """Categories must have distinct circadian shapes."""
    top = txns.mcc.value_counts().head(5).index
    dists = []
    for k in top:
        h = np.bincount(txns[txns.mcc == k].hour, minlength=24)[:24].astype(float)
        dists.append(h / h.sum())
    spread = max(np.abs(a - b).sum() for a in dists for b in dists)
    assert spread > 0.2


def test_users_are_loyal_to_their_regulars(txns):
    """Reference: top-5 merchants carry ~46% of a user's transactions."""
    busy = txns.groupby("user_id").filter(lambda g: len(g) >= 50)
    loyalty = busy.groupby("user_id").merchant_id.apply(
        lambda s: s.value_counts().head(5).sum() / len(s)
    )
    assert 0.25 < loyalty.median() < 0.75


def test_users_have_home_geography(txns):
    """~83% of reference spend is in the user's modal state."""
    onsite = txns[txns.state != "ONLINE"]
    busy = onsite.groupby("user_id").filter(lambda g: len(g) >= 50)
    home = busy.groupby("user_id").state.apply(
        lambda s: s.value_counts().iloc[0] / len(s)
    )
    assert home.median() > 0.6


def test_user_spend_scale_is_persistent(txns):
    """A user's position in the amount distribution must be consistent, or
    per-entity behavioural features carry no signal."""
    busy = txns[txns.amount > 0].groupby("user_id").filter(lambda g: len(g) >= 60)
    med = busy.groupby("user_id").amount.median()
    assert med.quantile(0.9) / med.quantile(0.1) > 1.5


def test_round_amounts_are_present(txns):
    """Reference: 11.0% of amounts land on a whole unit. Smooth draws sit at
    1% and are separable on that alone."""
    a = txns.amount[txns.amount > 0]
    round_share = float(np.isclose(a, np.round(a)).mean())
    assert round_share > 0.04


def test_legit_population_carries_no_fraud(txns):
    """Attacks are injected on top; the base world must be clean."""
    assert txns.is_fraud.sum() == 0
    assert txns.attack_id.isna().all()
