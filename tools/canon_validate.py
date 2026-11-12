#!/usr/bin/env python3
"""
Validate canon node files.

Usage:
    uv run --package kinship-shared python tools/canon_validate.py canon/patterns/

Prints one line per node and exits non-zero on any validation error.
"""

from __future__ import annotations

import sys
from pathlib import Path

from kinship_shared.canon_loader import load_canon, load_node, node_paths


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    directory = Path(argv[1])
    if not directory.is_dir():
        print(f"FAIL {directory}: not a directory", file=sys.stderr)
        return 2

    failed = False
    # Per-file pass first so every broken node is reported, not just the first.
    for path in node_paths(directory):
        try:
            node = load_node(path)
        except ValueError as exc:
            failed = True
            print(f"FAIL {path.name}: {' '.join(str(exc).split())}")
            continue
        print(
            f"OK {node.id} ({node.status}, {len(node.edges)} edges, "
            f"{len(node.checks)} check{'' if len(node.checks) == 1 else 's'})"
        )

    if not failed:
        try:
            nodes = load_canon(directory)
        except ValueError as exc:
            failed = True
            print(f"FAIL {exc}")
        else:
            print(f"{len(nodes)} node{'' if len(nodes) == 1 else 's'} valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
