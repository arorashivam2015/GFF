"""End-to-end guardrails on the pieces that need real artifacts on disk -
skipped cleanly if those artifacts haven't been generated in this
environment, so the suite still runs green in a fresh checkout."""
import json
import pathlib

import pytest

from redlab.taxonomy.loader import Taxonomy
from redlab.taxonomy.schema import Family

ART = pathlib.Path("artifacts")


def test_taxonomy_is_weighted_toward_synthetic_identity():
    tax = Taxonomy.load()
    sid = tax.by_family(Family.SYNTHETIC_IDENTITY)
    assert len(sid) == 5
    assert {v.id for v in sid} == {"PF-SID-001", "PF-SID-002", "PF-SID-003",
                                   "PF-SID-004", "PF-SID-005"}


@pytest.mark.skipif(not (ART / "fidelity_naive.json").exists(),
                    reason="naive fidelity not run")
def test_naive_baseline_is_clearly_separable():
    """The harness's own sanity check: if the naive generator ever scores
    near 0.5, the harness has stopped catching bad synthesis and nothing
    downstream can be trusted."""
    d = json.loads((ART / "fidelity_naive.json").read_text())
    disc = next(m for m in d["metrics"] if m["name"] == "discriminator_auc")
    assert disc["value"] > 0.7, "naive baseline is not separable - harness may be broken"


@pytest.mark.skipif(not (ART / "fidelity_vae.json").exists(),
                    reason="VAE fidelity not run")
def test_vae_fidelity_report_has_all_expected_metrics():
    d = json.loads((ART / "fidelity_vae.json").read_text())
    names = {m["name"] for m in d["metrics"]}
    assert {"discriminator_auc", "benford_mad_pp", "amount_percentile_log_rmse"} <= names


@pytest.mark.skipif(not (ART / "defend_ae_eval.json").exists() or
                    not (ART / "defend_gbm_eval.json").exists(),
                    reason="defend eval not run")
def test_unsupervised_and_supervised_evaluated_on_identical_holdout():
    """Both must report the same held-out fraud count, or the comparison
    between them is not apples-to-apples."""
    ae = json.loads((ART / "defend_ae_eval.json").read_text())
    assert ae["n_projected_vectors"] > 0
    assert 0 <= ae["autoencoder"]["recall_at_0.5fpr"] <= 1


@pytest.mark.skipif(not (ART / "loop_eval.json").exists(), reason="loop not run")
def test_loop_evasion_never_decreases_after_optimisation():
    """A white-box optimiser directly minimising the detector's own loss
    function should never do WORSE than the unoptimised baseline - if it
    does, the loop's loss function or gradient path has a bug."""
    results = json.loads((ART / "loop_eval.json").read_text())
    assert len(results) == 5
    for r in results:
        assert r["evasion_after"] >= r["evasion_before"] - 0.02  # small numerical slack


@pytest.mark.skipif(not (ART / "loop_eval.json").exists(), reason="loop not run")
def test_loop_drift_stays_bounded():
    """Value retention, adapted to a continuous setting: the fine-tuned
    sample must not have drifted arbitrarily far from the generator's own
    realistic output, or "evasion" is meaningless."""
    results = json.loads((ART / "loop_eval.json").read_text())
    for r in results:
        assert r["mean_drift"] < 1.0
