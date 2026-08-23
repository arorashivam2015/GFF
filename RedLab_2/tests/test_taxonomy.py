"""Taxonomy integrity, including the network_role extension this solution adds."""
import pytest
from redlab.taxonomy.loader import Taxonomy
from redlab.taxonomy.schema import Family, NetworkRole


@pytest.fixture(scope="module")
def tax() -> Taxonomy:
    return Taxonomy.load()


def test_corpus_loads_and_validates(tax):
    assert len(tax) >= 40


def test_ids_are_unique(tax):
    ids = [v.id for v in tax]
    assert len(ids) == len(set(ids))


def test_every_family_is_populated(tax):
    for family in Family:
        assert tax.by_family(family), f"family {family.value} has no vectors"


def test_network_role_field_present_and_valid(tax):
    for v in tax:
        assert isinstance(v.network_role, NetworkRole)


def test_graph_motif_vectors_span_all_roles(tax):
    """The graph generator needs at least one vector per non-NONE role, or
    that motif type can never be rendered."""
    roles = {v.network_role for v in tax}
    for role in NetworkRole:
        if role == NetworkRole.NONE:
            continue
        assert role in roles, f"no vector carries network_role={role.value}"


def test_graph_tagged_vectors_are_a_minority_by_design(tax):
    """Most fraud has no defining multi-hop network shape; only vectors that
    genuinely do should be tagged, or the motif injector loses its signal."""
    tagged = sum(1 for v in tax if v.network_role != NetworkRole.NONE)
    assert 0 < tagged < len(tax) * 0.5
