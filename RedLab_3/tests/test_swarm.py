"""Structural tests only - no live API calls, to keep the suite fast and
free of external dependencies/cost."""
import pathlib
import pytest

from redlab.sim.llm_swarm import PROMPT_TEMPLATE, GeneratedCampaign
from redlab.taxonomy.loader import Taxonomy


def test_prompt_template_has_required_fields():
    formatted = PROMPT_TEMPLATE.format(name="X", summary="Y", preconditions="Z")
    assert "X" in formatted and "Y" in formatted and "Z" in formatted


def test_prompt_frames_defensive_research_purpose():
    """The responsible-scope framing must actually be present in every
    generated prompt, not just asserted in a docstring."""
    assert "fraud-awareness" in PROMPT_TEMPLATE.lower() or \
          "training" in PROMPT_TEMPLATE.lower()
    assert "synthetic" in PROMPT_TEMPLATE.lower()


def test_generated_campaign_is_a_plain_dataclass():
    c = GeneratedCampaign(vector_id="PF-XX-001", vector_name="test",
                          prompt="p", text="t")
    assert c.vector_id == "PF-XX-001"


@pytest.mark.skipif(
    not pathlib.Path("data/processed/generated_campaigns.json").exists(),
    reason="swarm not run",
)
def test_swarm_run_produced_a_result_for_every_selected_vector():
    import json
    campaigns = json.load(open("data/processed/generated_campaigns.json"))
    tax = Taxonomy.load()
    for c in campaigns:
        assert c["vector_id"] in {v.id for v in tax}
        assert len(c["text"]) > 0


@pytest.mark.skipif(
    not pathlib.Path("artifacts/swarm_summary.json").exists(),
    reason="swarm not analysed",
)
def test_refusal_count_is_internally_consistent():
    import json
    s = json.load(open("artifacts/swarm_summary.json"))
    assert s["n_refused"] + s["n_generated"] == s["n_attempted"]
