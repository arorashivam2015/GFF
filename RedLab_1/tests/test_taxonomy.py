"""Taxonomy integrity tests.

These are guardrails on the IDENTIFY pillar: they fail the build if the corpus
drifts into a state the downstream pillars cannot consume.
"""

import pytest

from redlab.taxonomy.loader import Taxonomy
from redlab.taxonomy.schema import Family, GenAIUplift, Maturity


@pytest.fixture(scope="module")
def tax() -> Taxonomy:
    return Taxonomy.load()


def test_corpus_loads_and_validates(tax):
    assert len(tax) >= 40, "corpus should carry at least 40 vectors for the diversity criterion"


def test_ids_are_unique(tax):
    ids = [v.id for v in tax]
    assert len(ids) == len(set(ids))


def test_every_family_is_populated(tax):
    for family in Family:
        assert tax.by_family(family), f"family {family.value} has no vectors"


def test_every_vector_has_genai_uplift(tax):
    """A vector with no GenAI uplift is classical fraud and out of scope."""
    for v in tax:
        assert v.genai_uplift, f"{v.id} declares no GenAI uplift"


def test_detection_hypotheses_cover_declared_signals(tax):
    """Enforced by the schema; asserted here so the guarantee is visible."""
    for v in tax:
        assert {h.channel for h in v.detection_hypotheses} >= set(v.signals), v.id


def test_simulation_ranges_are_ordered(tax):
    for v in tax:
        assert v.simulation.duration_days[0] <= v.simulation.duration_days[1], v.id
        assert v.simulation.events_per_campaign[0] <= v.simulation.events_per_campaign[1], v.id


def test_corpus_spans_maturity_levels(tax):
    """We must be able to distinguish observed fraud from projected fraud."""
    present = {v.maturity for v in tax}
    assert present == set(Maturity), f"missing maturity levels: {set(Maturity) - present}"


def test_holdout_families_are_disjoint_from_train(tax):
    holdout_families = [Family.AGENTIC_COMMERCE, Family.ANTI_DEFENSE]
    train, holdout = tax.holdout_split(holdout_families)
    assert len(train) + len(holdout) == len(tax)
    assert not ({v.id for v in train} & {v.id for v in holdout})
    assert all(v.family in holdout_families for v in holdout)


def test_frontier_uplifts_are_represented(tax):
    """Prompt injection and adaptive evasion are the novelty bets; if the
    corpus loses them, the submission loses its differentiation."""
    covered = {u for v in tax for u in v.genai_uplift}
    for required in (GenAIUplift.PROMPT_INJECTION, GenAIUplift.ADAPTIVE_EVASION,
                     GenAIUplift.ORCHESTRATION):
        assert required in covered, f"{required.value} is unrepresented"


def test_graph_signal_is_well_represented(tax):
    """Graph detection is a core defence bet - the corpus must exercise it."""
    graph_vectors = [v for v in tax if any(s.value == "graph" for s in v.signals)]
    assert len(graph_vectors) >= 15, "too few graph-observable vectors to justify a graph model"
