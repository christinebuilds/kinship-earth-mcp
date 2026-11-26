#!/usr/bin/env python3
"""
Canon CI: the checks a pattern pull request must pass (spec 021).

Usage:
    uv run --package kinship-orchestrator python tools/canon_ci.py canon/patterns/

Offline and deterministic. Prints one line per finding and exits non-zero
on any error. It does not run check runners; the nightly job does that.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from kinship_shared.canon_loader import load_canon, load_node, load_stubs, node_paths
from kinship_shared.canon_tools import _names

PRESENCE_ONLY_ADAPTERS = {"gbif", "inaturalist"}


def ci_findings(canon_dir: str | Path, repo_root: str | Path | None = None) -> list[str]:
    """Every problem found, as plain sentences. Empty list = pass."""
    canon_dir = Path(canon_dir)
    root = Path(repo_root) if repo_root else canon_dir.resolve().parent.parent
    errors: list[str] = []

    # 1. Schema, per file, then cross-node edges.
    for path in node_paths(canon_dir):
        try:
            load_node(path)
        except ValueError as exc:
            errors.append(f"{path.name}: {' '.join(str(exc).split())}")
    if errors:
        return errors
    try:
        nodes = load_canon(canon_dir)
        stubs = load_stubs(canon_dir)
    except ValueError as exc:
        return [" ".join(str(exc).split())]

    # 2. One pattern, one node.
    owners: dict[str, set[str]] = defaultdict(set)
    for node in nodes.values():
        for name in _names(node):
            owners[name].add(node.id)
    for name, ids in sorted(owners.items()):
        if len(ids) > 1:
            errors.append(f"duplicate name or alias '{name}' in nodes: {', '.join(sorted(ids))} "
                          "(one pattern, one node: merge them or drop the alias)")

    # 3. Stubs are honest.
    referenced = {e.to for n in nodes.values() for e in n.edges}
    for stub in sorted(stubs):
        if stub in nodes:
            errors.append(f"_stubs.yaml: '{stub}' is now a written node; remove it from the stubs")
        elif stub not in referenced:
            errors.append(f"_stubs.yaml: '{stub}' is not referenced by any edge")

    for node in nodes.values():
        for check in node.checks:
            # 4. Runners exist.
            if not (root / check.runner).exists():
                errors.append(f"{node.id}: check '{check.id}' runner not found: {check.runner}")
            # 5. Status matches stored results.
            if node.status == "supported" and check.result and not check.passed:
                errors.append(f"{node.id}: status is 'supported' but check '{check.id}' last failed; "
                              "flip to 'contested' and ask a reviewer")
            # 6. Presence-only data cannot test count-based patterns.
            adapter = str(check.data.get("adapter", "")).lower()
            needs_counts = any("count" in c.lower() for c in node.regime.conditions)
            if needs_counts and adapter in PRESENCE_ONLY_ADAPTERS:
                errors.append(f"{node.id}: check '{check.id}' uses presence-only source '{adapter}', "
                              "but this pattern requires true counts")
    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    directory = Path(argv[1])
    if not directory.is_dir():
        print(f"FAIL {directory}: not a directory", file=sys.stderr)
        return 2
    findings = ci_findings(directory)
    for f in findings:
        print(f"FAIL {f}")
    n = len(node_paths(directory))
    print(f"{'OK' if not findings else 'FAILED'}: {n} node{'s' if n != 1 else ''}, "
          f"{len(findings)} problem{'s' if len(findings) != 1 else ''}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
