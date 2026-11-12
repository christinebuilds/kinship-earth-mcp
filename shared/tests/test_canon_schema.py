"""
Tests for the canon node schema, loader, and graph projection (spec 015).
"""

import shutil
from pathlib import Path

import frontmatter
import pytest

from kinship_shared.canon_loader import canon_to_graph, load_canon, load_node
from kinship_shared.canon_schema import CanonNode
from kinship_shared.graph_store import EcologicalGraph

FIXTURES = Path(__file__).parent / "fixtures" / "canon"


def _fixture_meta() -> dict:
    return dict(frontmatter.load(str(FIXTURES / "example-law.md")).metadata)


def test_fixture_node_loads_and_validates():
    node = load_node(FIXTURES / "example-law.md")
    assert node.id == "example-law"
    assert node.status == "supported"
    assert node.care_status == "public"
    assert len(node.evidence) == 3
    assert node.checks[0].passed is False
    assert node.edges[1].confidence == "conjectural"
    assert node.narrative.startswith("A fixture node")

    nodes = load_canon(FIXTURES)
    assert list(nodes) == ["example-law"]


def test_supported_needs_check_or_two_evidence():
    meta = _fixture_meta()
    meta["checks"] = []
    meta["evidence"] = [{"type": "paper", "ref": "Nobody 1900", "role": "origin"}]
    with pytest.raises(ValueError, match="supported"):
        CanonNode(**meta)

    # A passing check alone is enough.
    meta["checks"] = [{
        "id": "c", "runner": "canon/checks/example.py",
        "result": {"pass": True, "b": 1.5},
    }]
    assert CanonNode(**meta).checks[0].passed


def test_open_problem_rules():
    meta = _fixture_meta()
    meta["kind"] = "open_problem"
    meta["status"] = "conjecture"
    with pytest.raises(ValueError, match="must not carry checks"):
        CanonNode(**meta)

    meta["checks"] = []
    assert CanonNode(**meta).kind == "open_problem"

    meta["evidence"] = meta["evidence"][:2]
    with pytest.raises(ValueError, match="three evidence"):
        CanonNode(**meta)


def test_dangling_edge_needs_stub(tmp_path):
    shutil.copy(FIXTURES / "example-law.md", tmp_path / "example-law.md")
    with pytest.raises(ValueError, match="example-stub"):
        load_canon(tmp_path)

    shutil.copy(FIXTURES / "_stubs.yaml", tmp_path / "_stubs.yaml")
    assert "example-law" in load_canon(tmp_path)


def test_invalid_id_and_edge_type_rejected():
    meta = _fixture_meta()
    with pytest.raises(ValueError):
        CanonNode(**{**meta, "id": "Example_Law"})
    bad_edges = [{"to": "example-stub", "type": "INFORMS"}]
    with pytest.raises(ValueError):
        CanonNode(**{**meta, "edges": bad_edges})


@pytest.mark.asyncio
async def test_canon_to_graph(tmp_path):
    graph = EcologicalGraph(db_path=str(tmp_path / "g.db"))
    await graph.initialize()
    await canon_to_graph(load_canon(FIXTURES), graph)

    patterns = await graph.find_entities("pattern")
    assert [p.id for p in patterns] == ["pattern:example-law"]
    assert patterns[0].properties["status"] == "supported"

    sources = await graph.find_entities("data_source")
    assert [s.id for s in sources] == ["source:neonscience"]

    rels = await graph.get_relationships("pattern:example-law")
    by_type = sorted(r.relationship_type for r in rels)
    # Two edges share a target; the DiGraph keeps one edge per node pair.
    assert "EVIDENCED_BY" in by_type
    assert len(rels) == 2

    # Survives a save/load round trip.
    await graph.save()
    reloaded = EcologicalGraph(db_path=str(tmp_path / "g.db"))
    await reloaded.initialize()
    assert (await reloaded.get_entity("pattern:example-law")).entity_type == "pattern"
