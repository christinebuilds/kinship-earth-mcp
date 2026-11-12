"""
Load canon node files and project them into the knowledge graph.

``load_node``  — one Markdown file → validated ``CanonNode``
``load_canon`` — a directory → ``{id: CanonNode}``, with cross-node edge checks
``canon_to_graph`` — nodes → ``pattern`` entities + evidential edges
"""

from __future__ import annotations

from pathlib import Path

import frontmatter
import yaml

from .canon_schema import CanonNode
from .graph_schema import (
    GraphEntity,
    GraphRelationship,
    make_pattern_id,
    make_source_id,
)
from .graph_store import EcologicalGraph

STUBS_FILENAME = "_stubs.yaml"


class CanonValidationError(ValueError):
    """A canon directory failed cross-node validation."""


def load_node(path: str | Path) -> CanonNode:
    """Parse one node file. Raises ``ValueError`` if it is invalid."""
    post = frontmatter.load(str(path))
    if not post.metadata:
        raise CanonValidationError(f"{path}: no YAML front matter")
    narrative = post.content.strip() or None
    return CanonNode(**post.metadata, narrative=narrative)


def load_stubs(directory: str | Path) -> set[str]:
    """Placeholder ids for nodes not yet written (``_stubs.yaml``, a YAML list)."""
    path = Path(directory) / STUBS_FILENAME
    if not path.exists():
        return set()
    data = yaml.safe_load(path.read_text()) or []
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise CanonValidationError(f"{path}: expected a YAML list of node ids")
    return set(data)


def node_paths(directory: str | Path) -> list[Path]:
    """Node files in a directory (``*.md``, skipping README and ``_``-prefixed)."""
    return sorted(
        p for p in Path(directory).glob("*.md")
        if not p.name.startswith("_") and p.name.lower() != "readme.md"
    )


def load_canon(directory: str | Path) -> dict[str, CanonNode]:
    """Load every node in a directory, then validate edges across nodes."""
    nodes: dict[str, CanonNode] = {}
    for path in node_paths(directory):
        node = load_node(path)
        if node.id != path.stem:
            raise CanonValidationError(
                f"{path.name}: id '{node.id}' does not match filename"
            )
        if node.id in nodes:
            raise CanonValidationError(f"duplicate node id '{node.id}'")
        nodes[node.id] = node

    known = set(nodes) | load_stubs(directory)
    for node in nodes.values():
        for edge in node.edges:
            if edge.to not in known:
                raise CanonValidationError(
                    f"{node.id}: edge target '{edge.to}' is not a node id and is "
                    f"not listed in {STUBS_FILENAME}"
                )
    return nodes


async def canon_to_graph(nodes: dict[str, CanonNode], store: EcologicalGraph) -> None:
    """Upsert nodes as ``pattern`` entities and their edges as relationships.

    Dataset evidence becomes an ``EVIDENCED_BY`` edge to a ``data_source`` entity.
    """
    for node in nodes.values():
        pattern_id = make_pattern_id(node.id)
        await store.add_entity(GraphEntity(
            id=pattern_id,
            entity_type="pattern",
            name=node.name,
            properties={
                "kind": node.kind,
                "status": node.status,
                "layer": node.layer,
                "aliases": node.aliases,
                "statement_plain": node.statement_plain,
                "care_status": node.care_status,
            },
        ))

        for edge in node.edges:
            props = {"confidence": edge.confidence}
            if edge.field:
                props["field"] = edge.field
            if edge.note:
                props["note"] = edge.note
            await store.add_relationship(GraphRelationship(
                source_id=pattern_id,
                target_id=make_pattern_id(edge.to),
                relationship_type=edge.type,
                properties=props,
            ))

        for ev in node.evidence:
            if ev.type != "dataset" or not ev.source:
                continue
            source_id = make_source_id(ev.source)
            await store.add_entity(GraphEntity(
                id=source_id, entity_type="data_source", name=ev.source,
            ))
            props = {"role": ev.role}
            if ev.product:
                props["product"] = ev.product
            await store.add_relationship(GraphRelationship(
                source_id=pattern_id,
                target_id=source_id,
                relationship_type="EVIDENCED_BY",
                properties=props,
            ))
