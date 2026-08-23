"""Structural guardrails on the graph-native GENERATE pillar."""
import pathlib
import numpy as np
import pandas as pd
import pytest

from redlab.sim.world import PaymentWorld, WorldConfig
from redlab.sim.conditionals import ConditionalProfile
from redlab.sim.fraud_profile import FraudProfile
from redlab.sim.graph_attacks import MotifInjector, MotifConfig, to_graph_schema
from redlab.taxonomy.loader import Taxonomy
from redlab.taxonomy.schema import NetworkRole

NEEDED = ["data/processed/conditional_profile.json", "data/processed/fraud_profile.json"]
pytestmark = pytest.mark.skipif(
    not all(pathlib.Path(p).exists() for p in NEEDED),
    reason="calibration profiles not built",
)


@pytest.fixture(scope="module")
def combined():
    cfg = WorldConfig(n_cardholders=1200, n_merchants=30000, days=90, seed=11,
                      cold_outreach_share=0.55)
    world = PaymentWorld(ConditionalProfile.load(NEEDED[0]), cfg)
    legit = world.generate()
    tax = Taxonomy.load()
    fp = FraudProfile.load(NEEDED[1])
    inj = MotifInjector(world, tax, fp, cfg=MotifConfig(seed=11))
    df, campaigns = inj.inject(legit)
    return df, campaigns, legit


def test_legit_graph_schema_has_src_dst(combined):
    df, _, legit = combined
    legit_g = to_graph_schema(legit)
    assert (legit_g.src_id == legit_g.user_id).all()
    assert (legit_g.dst_id == legit_g.merchant_id).all()


def test_every_graph_tagged_vector_is_rendered(combined):
    df, campaigns, _ = combined
    tax = Taxonomy.load()
    expected = {v.id for v in tax if v.network_role != NetworkRole.NONE}
    rendered = set(df[df.is_fraud == 1].attack_id.unique())
    assert expected <= rendered, f"missing: {expected - rendered}"


def test_layering_chains_conserve_flow(combined):
    df, _, _ = combined
    chains = df[(df.is_fraud == 1) & (df.network_role == "layering_hop")]
    for mid, g in chains.groupby("motif_id"):
        g = g.sort_values("hop_index")
        amts = g.amount.to_numpy()
        if len(amts) > 1:
            assert np.all(np.diff(amts) <= 1e-6), f"{mid} does not conserve flow"


def test_fan_motifs_have_single_hub(combined):
    df, _, _ = combined
    f = df[df.is_fraud == 1]
    for role, hub_col in [("fan_out", "src_id"), ("fan_in", "dst_id")]:
        sub = f[f.network_role == role]
        if not len(sub):
            continue
        sizes = sub.groupby("motif_id")[hub_col].nunique()
        assert (sizes == 1).all()


def test_mule_nodes_are_fresh_not_reused_from_legit_population(combined):
    """Mule/attacker node IDs must not collide with real user or merchant
    IDs, or graph structure would silently merge attack and legit entities."""
    df, _, legit = combined
    real_ids = set(legit.user_id) | set(legit.merchant_id)
    mule_ids = set(df.loc[df.dst_type == "mule", "dst_id"]) | set(
        df.loc[df.src_type == "attacker", "src_id"])
    assert not (real_ids & mule_ids)


def test_combined_graph_has_no_fraud_in_legit_rows(combined):
    df, _, _ = combined
    assert (df[df.network_role == "none"].is_fraud == 0).all()
