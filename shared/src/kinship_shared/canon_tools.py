"""
Read-only access to the canon for MCP tools (spec 020).

Pure functions over ``canon/patterns``: list, get, neighbours, check status,
and validating a proposed node. Nothing here writes files, runs check
runners, or changes a node's status.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import frontmatter
import yaml

from .canon_loader import load_canon, load_node, load_stubs, node_paths
from .canon_schema import CanonNode

_REPO_CANON = Path(__file__).resolve().parents[3] / "canon" / "patterns"
_cache: dict[str, tuple[tuple, dict[str, CanonNode], set[str]]] = {}


def default_canon_dir() -> Path:
    return Path(os.environ.get("KINSHIP_CANON_DIR") or _REPO_CANON)


def _signature(directory: Path) -> tuple:
    files = list(node_paths(directory)) + [directory / "_stubs.yaml"]
    return tuple((p.name, p.stat().st_mtime_ns) for p in files if p.exists())


def _load(canon_dir: Optional[Path]) -> tuple[dict[str, CanonNode], set[str], Path]:
    directory = Path(canon_dir) if canon_dir else default_canon_dir()
    key = str(directory.resolve())
    sig = _signature(directory)
    hit = _cache.get(key)
    if hit is None or hit[0] != sig:
        _cache[key] = (sig, load_canon(directory), load_stubs(directory))
    _, nodes, stubs = _cache[key]
    return nodes, stubs, directory


def _norm(text: str) -> str:
    """'Taylor's law' -> 'taylors law'; 'taylors-power-law' -> 'taylors power law'."""
    text = text.lower().replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _names(node: CanonNode) -> set[str]:
    return {_norm(node.id), _norm(node.name), *(_norm(a) for a in node.aliases)}


def _summary(node: CanonNode) -> dict:
    last = node.checks[0] if node.checks else None
    return {
        "id": node.id,
        "name": node.name,
        "kind": node.kind,
        "status": node.status,
        "layer": node.layer,
        "statement_plain": " ".join(node.statement_plain.split()),
        "n_checks": len(node.checks),
        "last_check_pass": (last.passed if last and last.result else None),
    }


def canon_list(status: str | None = None, kind: str | None = None,
               layer: str | None = None, canon_dir: Path | None = None) -> list[dict]:
    nodes, _, _ = _load(canon_dir)
    out = []
    for node in sorted(nodes.values(), key=lambda n: n.id):
        if status and node.status != status:
            continue
        if kind and node.kind != kind:
            continue
        if layer and node.layer != layer:
            continue
        out.append(_summary(node))
    return out


def _resolve(nodes: dict[str, CanonNode], query: str) -> Optional[CanonNode]:
    if query in nodes:
        return nodes[query]
    q = _norm(query)
    exact = [n for n in nodes.values() if q in _names(n)]
    if exact:
        return exact[0]
    # "taylors law" should find "taylors power law": every query word present.
    words = set(q.split())
    loose = [n for n in nodes.values() if words and any(words <= set(x.split()) for x in _names(n))]
    return loose[0] if len(loose) == 1 else None


def canon_get(node_id_or_name: str, canon_dir: Path | None = None) -> dict | None:
    nodes, _, directory = _load(canon_dir)
    node = _resolve(nodes, node_id_or_name)
    if node is None:
        return None
    data = node.model_dump(mode="json")
    path = directory / f"{node.id}.md"
    data["file"] = (f"canon/patterns/{node.id}.md" if directory.resolve() == _REPO_CANON.resolve()
                    else str(path))
    return data


def canon_neighbors(node_id: str, depth: int = 1, canon_dir: Path | None = None) -> dict:
    nodes, stubs, _ = _load(canon_dir)
    start = _resolve(nodes, node_id)
    if start is None:
        return {"id": node_id, "found": False, "edges": []}
    seen, frontier, edges = {start.id}, [start.id], []
    for level in range(1, max(1, depth) + 1):
        nxt = []
        for nid in frontier:
            outgoing = nodes[nid].edges if nid in nodes else []
            for e in outgoing:
                edges.append({"from": nid, "to": e.to, "type": e.type, "field": e.field,
                              "note": e.note, "direction": "out", "depth": level,
                              "to_is_stub": e.to not in nodes and e.to in stubs})
                if e.to not in seen:
                    seen.add(e.to)
                    nxt.append(e.to)
            for other in nodes.values():
                for e in other.edges:
                    if e.to == nid and other.id != nid:
                        edges.append({"from": other.id, "to": nid, "type": e.type, "field": e.field,
                                      "note": e.note, "direction": "in", "depth": level,
                                      "to_is_stub": False})
                        if other.id not in seen:
                            seen.add(other.id)
                            nxt.append(other.id)
        frontier = nxt
    return {"id": start.id, "found": True, "edges": edges}


def canon_check_status(node_id: str, canon_dir: Path | None = None) -> list[dict]:
    """Stored check results only. Never imports or runs a check runner."""
    nodes, _, _ = _load(canon_dir)
    node = _resolve(nodes, node_id)
    if node is None:
        return []
    return [{
        "id": c.id, "runner": c.runner, "expect": c.expect,
        "last_run": c.last_run.isoformat() if c.last_run else None,
        "result": c.result, "passed": c.passed if c.result else None,
    } for c in node.checks]


def canon_propose(node_yaml: str, narrative: str = "", canon_dir: Path | None = None) -> dict:
    """Validate a proposed node. Returns the file text it would have; writes nothing."""
    out: dict = {"valid": False, "errors": [], "node_id": None, "file_path": None,
                 "file_text": None, "duplicates": []}
    try:
        meta = yaml.safe_load(node_yaml) or {}
        if not isinstance(meta, dict):
            raise ValueError("proposal must be a YAML mapping of node fields")
        node = CanonNode(**meta, narrative=narrative.strip() or None)
    except Exception as exc:  # pydantic ValidationError, YAML errors
        out["errors"] = [" ".join(str(exc).split())]
        return out

    nodes, stubs, _ = _load(canon_dir)
    proposed = _names(node)
    out["duplicates"] = sorted(n.id for n in nodes.values() if _names(n) & proposed)
    missing = [e.to for e in node.edges if e.to not in nodes and e.to not in stubs]
    if missing:
        out["errors"].append(
            "edge targets not in the canon or _stubs.yaml (add them as stubs): " + ", ".join(missing))

    post = frontmatter.Post(narrative.strip(), **meta)
    text = frontmatter.dumps(post) + "\n"
    out.update(node_id=node.id, file_path=f"canon/patterns/{node.id}.md", file_text=text)
    out["valid"] = not out["errors"] and not out["duplicates"]
    if out["duplicates"]:
        out["errors"].append("one pattern, one node: add aliases or edges to the existing node instead")
    return out
