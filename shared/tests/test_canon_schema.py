"""
Tests for canon schema, loader, and graph integration (spec 017).

All tests are offline — no network calls.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from kinship_shared.canon_loader import load_canon, load_node
from kinship_shared.canon_schema import CanonNode
from kinship_shared.graph_store import EcologicalGraph

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "canon"


# ---------------------------------------------------------------------------
# 1. Fixture node loads and validates
# ---------------------------------------------------------------------------

def test_fixture_node_loads():
    node = load_node(FIXTURE_DIR / "example-law.md")
    assert node.id == "example-law"
    assert node.status == "supported"
    assert node.kind == "empirical_law"
    assert len(node.evidence) == 3
    assert len(node.edges) == 1
    assert node.edges[0].to == "stub-node-a"
    assert node.narrative is not None


# ---------------------------------------------------------------------------
# 2. status == "supported" with no check and one evidence entry → error
# ---------------------------------------------------------------------------

def test_supported_requires_two_origin_support_evidence():
    with pytest.raises(ValueError, match="supported"):
        CanonNode.model_validate({
            "id": "bad-node",
            "name": "Bad Node",
            "kind": "empirical_law",
            "status": "supported",
            "layer": "population",
            "statement_plain": "A node that should fail validation.",
            "regime": {"taxa": ["all"], "scales": [], "conditions": []},
            "evidence": [
                {"type": "paper", "doi": "10.0/x", "ref": "X 2000", "role": "origin"},
            ],
            "checks": [],
        })


# ---------------------------------------------------------------------------
# 3. kind == "open_problem" with a check → error
# ---------------------------------------------------------------------------

def test_open_problem_must_have_no_checks():
    with pytest.raises(ValueError, match="checks"):
        CanonNode.model_validate({
            "id": "open-with-check",
            "name": "Open Problem With Check",
            "kind": "open_problem",
            "status": "conjecture",
            "layer": "community",
            "statement_plain": "An open problem that erroneously has a check.",
            "regime": {},
            "evidence": [
                {"type": "paper", "doi": "10.0/a", "ref": "A 2000", "role": "origin"},
                {"type": "paper", "doi": "10.0/b", "ref": "B 2000", "role": "support"},
                {"type": "paper", "doi": "10.0/c", "ref": "C 2000", "role": "support"},
            ],
            "checks": [
                {"id": "bad-check", "runner": "canon/checks/bad.py", "data": {}, "expect": {}}
            ],
        })


# ---------------------------------------------------------------------------
# 4. Dangling edge target not in stubs → error; in stubs → OK
# ---------------------------------------------------------------------------

def test_dangling_edge_raises(tmp_path):
    node_dir = tmp_path / "patterns"
    node_dir.mkdir()
    (node_dir / "example-law.md").write_text(
        (FIXTURE_DIR / "example-law.md").read_text()
    )
    # No _stubs.yaml → stub-node-a is unknown → should raise
    with pytest.raises(ValueError, match="stub-node-a"):
        load_canon(node_dir)


def test_edge_to_stub_ok(tmp_path):
    node_dir = tmp_path / "patterns"
    node_dir.mkdir()
    (node_dir / "example-law.md").write_text(
        (FIXTURE_DIR / "example-law.md").read_text()
    )
    (node_dir / "_stubs.yaml").write_text("- stub-node-a\n")
    nodes = load_canon(node_dir)
    assert "example-law" in nodes


# ---------------------------------------------------------------------------
# 5. canon_to_graph creates one pattern entity and correct relationships
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_canon_to_graph(tmp_path):
    from kinship_shared.canon_loader import canon_to_graph

    nodes = load_canon(FIXTURE_DIR)

    db_path = str(tmp_path / "test.db")
    store = EcologicalGraph(db_path=db_path)
    await store.initialize()

    await canon_to_graph(nodes, store)

    # One pattern entity for example-law
    entity = await store.get_entity("pattern:example-law")
    assert entity is not None
    assert entity.entity_type == "pattern"
    assert entity.name == "Example Law"

    # One ANALOG_OF edge (stub-node-a) + one EVIDENCED_BY edge (neon dataset)
    rels = await store.get_relationships("pattern:example-law")
    rel_types = {r.relationship_type for r in rels}
    assert "ANALOG_OF" in rel_types
    assert "EVIDENCED_BY" in rel_types


# ---------------------------------------------------------------------------
# 6. Existing graph tests still pass after the ontology change
#    (this is verified by running the full suite; here we smoke-test EntityType)
# ---------------------------------------------------------------------------

def test_pattern_entity_type_in_ontology():
    from kinship_shared.graph_schema import EntityType
    from typing import get_args
    assert "pattern" in get_args(EntityType)


def test_new_relationship_types_in_ontology():
    from kinship_shared.graph_schema import RelationshipType
    from typing import get_args
    new_types = {
        "DERIVES_FROM", "GENERALIZES", "SPECIAL_CASE_OF",
        "SUPPORTS", "CONTRADICTS", "REQUIRES",
        "ANALOG_OF", "TESTED_BY", "EVIDENCED_BY",
    }
    existing = set(get_args(RelationshipType))
    assert new_types.issubset(existing)
