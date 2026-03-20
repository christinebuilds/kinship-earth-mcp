#!/usr/bin/env python3
"""
Schema Snapshot Tool — detects when upstream APIs add or remove fields.

Makes one canonical query per adapter, extracts the raw response schema
(field names + types), saves a snapshot, and diffs against the previous
snapshot to surface new fields (opportunity), missing fields (risk),
and type changes (risk).

Usage:
    python tools/schema_snapshot.py              # run all adapters
    python tools/schema_snapshot.py --adapters obis,gbif   # run specific ones
    python tools/schema_snapshot.py --dry-run    # print schema without saving

Snapshots are saved to:
    ~/Coding/notes/api-health-logs/snapshots/YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SNAPSHOT_DIR = Path.home() / "Coding" / "notes" / "api-health-logs" / "snapshots"

# Each adapter entry defines:
#   url:    the API endpoint to probe
#   params: query params that return a small but representative response
#   path:   JSON path to the list of result records (dot-separated)
#   note:   human-readable description of the canonical query
#   auth_env: optional env var name for API key (skipped if missing)
#   auth_header: header name for the API key
ADAPTERS: dict[str, dict[str, Any]] = {
    "obis": {
        "url": "https://api.obis.org/v3/occurrence",
        "params": {"scientificname": "Tursiops truncatus", "size": 2},
        "path": "results",
        "note": "Bottlenose dolphins (global, 2 records)",
    },
    "inaturalist": {
        "url": "https://api.inaturalist.org/v1/observations",
        "params": {
            "taxon_name": "Calypte anna",
            "lat": 36.6,
            "lng": -121.9,
            "radius": 50,
            "per_page": 2,
            "order_by": "observed_on",
        },
        "path": "results",
        "note": "Anna's Hummingbird near Monterey (2 records)",
    },
    "gbif": {
        "url": "https://api.gbif.org/v1/occurrence/search",
        "params": {
            "scientificName": "Tursiops truncatus",
            "limit": 2,
            "hasCoordinate": "true",
        },
        "path": "results",
        "note": "Bottlenose dolphins with coordinates (2 records)",
    },
    "neon": {
        "url": "https://data.neonscience.org/api/v0/sites/WREF",
        "params": {},
        "path": "data",
        "note": "Wind River Experimental Forest site metadata",
        "single_record": True,
    },
    "era5": {
        "url": "https://archive-api.open-meteo.com/v1/archive",
        "params": {
            "latitude": 36.6,
            "longitude": -121.9,
            "start_date": "2025-01-01",
            "end_date": "2025-01-01",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "models": "era5",
            "timezone": "UTC",
        },
        "path": "hourly",
        "note": "ERA5 hourly climate for Monterey on 2025-01-01",
        "single_record": True,
    },
    "usgs_nwis": {
        "url": "https://waterservices.usgs.gov/nwis/iv/",
        "params": {
            "format": "json",
            "sites": "11143000",
            "parameterCd": "00060",
            "siteStatus": "active",
        },
        "path": "value.timeSeries",
        "note": "USGS site 11143000 (Big Sur River) — instantaneous streamflow",
    },
    "xeno_canto": {
        "url": "https://xeno-canto.org/api/3/recordings",
        "params": {"query": "Turdus migratorius", "page": 1},
        "path": "recordings",
        "note": "American Robin recordings (API v3, requires XC_API_KEY)",
        "auth_env": "XC_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "soilgrids": {
        "url": "https://rest.isric.org/soilgrids/v2.0/properties/query",
        "params": {
            "lat": 36.6,
            "lon": -121.9,
            "property": "soc",
            "depth": "0-5cm",
            "value": "mean",
        },
        "path": "properties.layers",
        "note": "Soil organic carbon at Monterey, 0-5cm depth",
    },
    "ebird": {
        "url": "https://api.ebird.org/v2/data/obs/geo/recent",
        "params": {"lat": 36.6, "lng": -121.9, "dist": 10, "maxResults": 2},
        "note": "Recent observations near Monterey (2 records)",
        "path": "",  # eBird returns a top-level list
        "auth_env": "EBIRD_API_KEY",
        "auth_header": "X-eBirdApiToken",
    },
}


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------


def _python_type(value: Any) -> str:
    """Return a stable type label for a JSON value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        if len(value) == 0:
            return "list[empty]"
        # Describe the element type of the first item
        return f"list[{_python_type(value[0])}]"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def extract_schema(data: Any) -> dict[str, str]:
    """
    Extract a flat field→type mapping from a JSON object or list of objects.

    For a list of records, unions the field sets across all records so we
    capture optional fields that only appear on some records.
    """
    if isinstance(data, list):
        merged: dict[str, str] = {}
        for record in data:
            if isinstance(record, dict):
                for key, val in record.items():
                    type_label = _python_type(val)
                    # Keep the most informative type (non-null wins)
                    if key not in merged or merged[key] == "null":
                        merged[key] = type_label
        return dict(sorted(merged.items()))
    elif isinstance(data, dict):
        return dict(sorted((k, _python_type(v)) for k, v in data.items()))
    else:
        return {"_root": _python_type(data)}


def _resolve_path(obj: Any, path: str) -> Any:
    """Walk a dot-separated path into a nested dict. Empty path returns obj."""
    if not path:
        return obj
    for segment in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(segment)
        else:
            return None
    return obj


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


def diff_schemas(
    old: dict[str, str], new: dict[str, str]
) -> dict[str, list[dict[str, str]]]:
    """
    Compare two field→type mappings.

    Returns:
        {
            "added":   [{"field": "x", "type": "str"}],
            "removed": [{"field": "y", "type": "int"}],
            "changed": [{"field": "z", "old_type": "str", "new_type": "int"}],
        }
    """
    old_fields = set(old.keys())
    new_fields = set(new.keys())

    added = [{"field": f, "type": new[f]} for f in sorted(new_fields - old_fields)]
    removed = [{"field": f, "type": old[f]} for f in sorted(old_fields - new_fields)]
    changed = []
    for f in sorted(old_fields & new_fields):
        if old[f] != new[f]:
            # null→anything is not a real change (field was previously unseen)
            if old[f] == "null" or new[f] == "null":
                continue
            changed.append({"field": f, "old_type": old[f], "new_type": new[f]})

    return {"added": added, "removed": removed, "changed": changed}


# ---------------------------------------------------------------------------
# API probing
# ---------------------------------------------------------------------------


import os


def _fetch_adapter(name: str, config: dict[str, Any], timeout: float = 30) -> dict:
    """
    Probe one adapter API and return its schema dict.
    Raises on network or parse errors (caller handles gracefully).
    """
    headers: dict[str, str] = {"Accept": "application/json"}

    # Auth handling
    auth_env = config.get("auth_env")
    if auth_env:
        key = os.environ.get(auth_env)
        if not key:
            raise EnvironmentError(
                f"Skipping {name}: env var {auth_env} not set"
            )
        auth_header = config.get("auth_header", "Authorization")
        auth_prefix = config.get("auth_prefix", "")
        headers[auth_header] = f"{auth_prefix}{key}"

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(config["url"], params=config.get("params", {}), headers=headers)
        resp.raise_for_status()
        body = resp.json()

    # Navigate to the relevant portion of the response
    data = _resolve_path(body, config.get("path", ""))
    if data is None:
        raise ValueError(f"Path '{config.get('path')}' resolved to None in response")

    # For single-record endpoints, wrap in a list for consistent handling
    if config.get("single_record") and isinstance(data, dict):
        data = [data]

    return extract_schema(data)


# ---------------------------------------------------------------------------
# Snapshot I/O
# ---------------------------------------------------------------------------


def save_snapshot(snapshot: dict, snapshot_date: str) -> Path:
    """Save a snapshot JSON file. Returns the file path."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = SNAPSHOT_DIR / f"{snapshot_date}.json"
    filepath.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    return filepath


def load_latest_snapshot(before_date: str) -> dict | None:
    """Load the most recent snapshot before the given date."""
    if not SNAPSHOT_DIR.exists():
        return None

    candidates = sorted(SNAPSHOT_DIR.glob("*.json"), reverse=True)
    for path in candidates:
        snap_date = path.stem
        if snap_date < before_date:
            return json.loads(path.read_text())

    return None


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------


def format_report(
    snapshot: dict,
    diffs: dict[str, dict] | None,
    errors: dict[str, str],
    snapshot_date: str,
) -> str:
    """Build a human-readable summary of the snapshot run."""
    lines: list[str] = []
    lines.append(f"Schema Snapshot Report — {snapshot_date}")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    ok_count = len(snapshot)
    err_count = len(errors)
    lines.append(f"Adapters probed: {ok_count + err_count}")
    lines.append(f"  Succeeded: {ok_count}")
    lines.append(f"  Failed:    {err_count}")
    lines.append("")

    # Errors
    if errors:
        lines.append("ERRORS")
        lines.append("-" * 40)
        for name, msg in sorted(errors.items()):
            lines.append(f"  {name}: {msg}")
        lines.append("")

    # Field counts per adapter
    lines.append("FIELD COUNTS")
    lines.append("-" * 40)
    for name in sorted(snapshot):
        count = len(snapshot[name])
        lines.append(f"  {name}: {count} fields")
    lines.append("")

    # Diffs
    if diffs:
        has_changes = any(
            d["added"] or d["removed"] or d["changed"]
            for d in diffs.values()
        )
        if not has_changes:
            lines.append("No schema changes detected since last snapshot.")
        else:
            lines.append("SCHEMA CHANGES (vs. previous snapshot)")
            lines.append("-" * 40)
            for name in sorted(diffs):
                d = diffs[name]
                if not d["added"] and not d["removed"] and not d["changed"]:
                    continue
                lines.append(f"\n  [{name}]")
                for item in d["added"]:
                    lines.append(
                        f"    + NEW FIELD: {item['field']} ({item['type']})"
                    )
                for item in d["removed"]:
                    lines.append(
                        f"    - MISSING FIELD: {item['field']} (was {item['type']})"
                    )
                for item in d["changed"]:
                    lines.append(
                        f"    ~ TYPE CHANGE: {item['field']}  "
                        f"{item['old_type']} -> {item['new_type']}"
                    )
    else:
        lines.append("No previous snapshot found — this is the baseline.")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot upstream API schemas and detect field changes."
    )
    parser.add_argument(
        "--adapters",
        type=str,
        default=None,
        help="Comma-separated list of adapter names to probe (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schemas without saving a snapshot file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="HTTP timeout per adapter in seconds (default: 30)",
    )
    args = parser.parse_args()

    today = date.today().isoformat()

    # Determine which adapters to probe
    if args.adapters:
        names = [n.strip() for n in args.adapters.split(",")]
        adapters = {n: ADAPTERS[n] for n in names if n in ADAPTERS}
        unknown = [n for n in names if n not in ADAPTERS]
        if unknown:
            print(f"Warning: unknown adapters ignored: {', '.join(unknown)}")
    else:
        adapters = ADAPTERS

    # Probe each adapter
    snapshot: dict[str, dict[str, str]] = {}
    errors: dict[str, str] = {}

    for name, config in sorted(adapters.items()):
        print(f"  Probing {name}... ", end="", flush=True)
        try:
            schema = _fetch_adapter(name, config, timeout=args.timeout)
            snapshot[name] = schema
            print(f"OK ({len(schema)} fields)")
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            errors[name] = error_msg
            print(f"FAILED — {error_msg}")

    # Diff against previous snapshot
    diffs: dict[str, dict] | None = None
    prev = load_latest_snapshot(today)
    if prev:
        diffs = {}
        for name in snapshot:
            old_schema = prev.get(name, {})
            diffs[name] = diff_schemas(old_schema, snapshot[name])

    # Report
    report = format_report(snapshot, diffs, errors, today)
    print()
    print(report)

    # Save (unless dry-run)
    if not args.dry_run and snapshot:
        filepath = save_snapshot(snapshot, today)
        print(f"Snapshot saved to: {filepath}")
    elif args.dry_run:
        print("(Dry run — snapshot not saved)")
        print()
        print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
