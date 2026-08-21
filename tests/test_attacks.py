"""Guardrails on attack realism.

Each assertion encodes a defect that was measured and fixed. The reference
values come from 28,619 labelled frauds in the reference corpus, so these are
calibration checks rather than taste.
"""

import pathlib

import numpy as np
import pytest

from redlab.sim.attacks import AttackConfig, inject_attacks
from redlab.sim.conditionals import ConditionalProfile
from redlab.sim.fraud_profile import FraudProfile
from redlab.sim.world import PaymentWorld, WorldConfig

NEEDED = ["data/processed/conditional_profile.json", "data/processed/fraud_profile.json"]
pytestmark = pytest.mark.skipif(
    not all(pathlib.Path(p).exists() for p in NEEDED),
    reason="profiles not built; see README quickstart",
)


@pytest.fixture(scope="module")
def attacked():
    cfg = WorldConfig(n_cardholders=1500, n_merchants=3000, days=120, seed=11)
    world = PaymentWorld(ConditionalProfile.load(NEEDED[0]), cfg)
    legit = world.generate()
    combined, campaigns = inject_attacks(world, legit, cfg=AttackConfig(seed=11))
    return combined, campaigns, legit


@pytest.fixture(scope="module")
def fp():
    return FraudProfile.load()


def test_every_vector_is_simulated(attacked):
    """The generic engine must render all 42 specs, not a hand-coded subset."""
    combined, _, _ = attacked
    assert combined.attack_id.nunique() == 42


def test_fraud_rate_is_near_target(attacked):
    combined, _, _ = attacked
    rate = combined.is_fraud.mean()
    assert 0.004 < rate < 0.03, f"fraud rate {rate:.4f} outside workable band"


def test_fraud_respects_victim_ceiling(attacked, fp):
    """Reference: only 0.74% of frauds exceed the victim's own historical max.

    An earlier version drew DRAIN amounts as a multiple of that max, putting
    40% of fraud above it, and the corpus became separable on amount alone.
    """
    combined, _, legit = attacked
    vmax = legit[legit.amount > 0].groupby("user_id").amount.max()
    fraud = combined[combined.is_fraud == 1]
    fraud = fraud[fraud.user_id.isin(vmax.index)]
    above = (fraud.amount.to_numpy() > vmax.reindex(fraud.user_id).to_numpy()).mean()
    assert above < 0.15, f"{above:.1%} of fraud exceeds victim max (reference 0.74%)"


def test_fraud_channel_mix_tracks_reference(attacked, fp):
    """Reference fraud is 61% online. 93% was an artefact of routing every
    transfer-rail vector through an online-only path."""
    combined, _, _ = attacked
    online = (combined[combined.is_fraud == 1].channel == "Online Transaction").mean()
    target = fp.channel_mix.get("Online Transaction", 0.61)
    assert abs(online - target) < 0.15, f"online share {online:.2f} vs reference {target:.2f}"


def test_fraud_spans_many_categories(attacked):
    """Reference fraud spans 98 MCCs with the top at 16.9%; collapsing onto a
    single money-transfer category made MCC separable at 0.89 on its own."""
    fraud = attacked[0][fraud_mask(attacked)]
    top_share = fraud.mcc.value_counts(normalize=True).iloc[0]
    assert fraud.mcc.nunique() >= 20
    assert top_share < 0.35, f"top MCC carries {top_share:.1%} of fraud"


def fraud_mask(attacked):
    return attacked[0].is_fraud == 1


def test_fraud_reuses_real_world_entities(attacked):
    """Fraud must hide inside the legitimate population, not beside it."""
    combined, _, legit = attacked
    fraud = combined[combined.is_fraud == 1]
    known = set(legit.merchant_id.unique())
    assert (fraud.merchant_id.isin(known)).mean() > 0.9
    known_users = set(legit.user_id.unique())
    assert (fraud.user_id.isin(known_users)).mean() > 0.99


def test_entity_reuse_shapes_campaign_concentration(attacked):
    """High-reuse specs must concentrate victims and low-reuse specs spread
    them, or the graph detector has nothing to distinguish."""
    _, campaigns, _ = attacked
    by_vec = {}
    for c in campaigns:
        by_vec.setdefault(c.vector_id, []).append(c.n_events / max(len(c.victim_users), 1))
    # PF-IND-005 is the highest-reuse spec in the corpus (0.85).
    dense = np.mean(by_vec.get("PF-IND-005", [1]))
    # PF-CT-005 is among the lowest (0.15).
    sparse = np.mean(by_vec.get("PF-CT-005", [1]))
    assert dense > sparse, f"reuse not honoured: dense={dense:.1f} sparse={sparse:.1f}"


def test_campaigns_have_coherent_timespans(attacked):
    _, campaigns, _ = attacked
    for c in campaigns[:200]:
        assert c.duration_days >= 1
        assert c.n_events >= 1
