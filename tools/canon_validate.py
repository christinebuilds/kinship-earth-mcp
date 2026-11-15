#!/usr/bin/env python
"""
Validate all canon node files in a directory.

Usage:
    uv run python tools/canon_validate.py canon/patterns/

Exits non-zero if any node fails validation.
Prints one line per node: OK <id> (<status>, <n> edges, <n> checks)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "src"))

from kinship_shared.canon_loader import load_canon


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <patterns-directory>", file=sys.stderr)
        return 1

    patterns_dir = Path(sys.argv[1])
    if not patterns_dir.is_dir():
        print(f"Not a directory: {patterns_dir}", file=sys.stderr)
        return 1

    errors: list[str] = []

    try:
        nodes = load_canon(patterns_dir)
    except ValueError as exc:
        print(f"ERROR (cross-node): {exc}", file=sys.stderr)
        return 1

    for node_id, node in sorted(nodes.items()):
        print(
            f"OK {node_id} ({node.status}, {len(node.edges)} edges, {len(node.checks)} checks)"
        )

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
