"""
Canon node loader: front matter → CanonNode → knowledge graph.

load_node   — parse one .md file into a validated CanonNode
load_canon  — load all nodes in a directory, with cross-node edge validation
canon_to_graph — upsert nodes and edges into an EcologicalGraph
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter
import yaml

from .canon_schema import CanonNode
from .graph_schema import GraphEntity, GraphRelationship

if TYPE_CHECKING:
    from .graph_store import EcologicalGraph


def load_node(path: str | Path) -> CanonNode:
    """Parse YAML front matter from a canon pattern file and return a validated node."""
    post = frontmatter.load(str(path))
    data = dict(post.metadata)
    body = post.content.strip()
    if body:
        data["narrative"] = body
    return CanonNode.model_validate(data)


def load_canon(directory: str | Path) -> dict[str, CanonNode]:
    """Load all canon nodes from a directory and validate cross-node edges.

    Stubs file: ``<directory>/_stubs.yaml`` lists placeholder node IDs for
    nodes not yet written. Edges pointing to stubs are permitted; edges
    pointing to unknown IDs raise ValueError.
    """
    directory = Path(directory)

    stubs_path = directory / "_stubs.yaml"
    stub_ids: set[str] = set()
    if stubs_path.exists():
        with open(stubs_path) as f:
            raw = yaml.safe_load(f)
            if raw:
                stub_ids = set(raw)

    nodes: dict[str, CanonNode] = {}
    for path in sorted(directory.glob("*.md")):
        node = load_node(path)
        nodes[node.id] = node

    all_known = set(nodes.keys()) | stub_ids
    for node in nodes.values():
        for edge in node.edges:
            if edge.to not in all_known:
                raise ValueError(
                    f"Node '{node.id}' has edge to unknown target '{edge.to}' "
                    f"(not in loaded nodes or {stubs_path})"
                )

    return nodes


async def canon_to_graph(
    nodes: dict[str, CanonNode],
    store: "EcologicalGraph",
) -> None:
    """Upsert canon nodes and their edges into the knowledge graph.

    Each canon node becomes a ``pattern`` entity.
    Pattern→pattern edges become graph relationships by their CanonEdgeType.
    Dataset evidence creates EVIDENCED_BY edges to data_source entities.
    """
    for node in nodes.values():
        entity = GraphEntity(
            id=f"pattern:{node.id}",
            entity_type="pattern",
            name=node.name,
            properties={
                "kind": node.kind,
                "status": node.status,
                "layer": node.layer,
                "statement_plain": node.statement_plain,
                "aliases": node.aliases,
            },
        )
        await store.add_entity(entity)

        for edge in node.edges:
            rel = GraphRelationship(
                source_id=f"pattern:{node.id}",
                target_id=f"pattern:{edge.to}",
                relationship_type=edge.type,
                properties={
                    "confidence": edge.confidence,
                    "field": edge.field,
                    "note": edge.note,
                },
            )
            await store.add_relationship(rel)

        for ev in node.evidence:
            if ev.type == "dataset" and ev.source:
                rel = GraphRelationship(
                    source_id=f"pattern:{node.id}",
                    target_id=f"source:{ev.source}",
                    relationship_type="EVIDENCED_BY",
                    properties={"product": ev.product, "role": ev.role},
                )
                await store.add_relationship(rel)
