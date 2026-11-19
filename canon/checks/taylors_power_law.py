#!/usr/bin/env python3
"""
Check: Taylor's power law on NEON breeding-landbird point counts (DP1.10003.001).

Taylor's power law: Var(N) = a * mean(N)^b, with b typically between 1 and 2.
A global exponent b in [1, 2] confirms the pipeline and the law.

Usage (from project root):
    uv run python canon/checks/taylors_power_law.py \\
        --sites WREF,HARV,SRER,OSBS,KONZ,CPER,TALL,BART,UNDE,JERC \\
        --year 2023

Output:
    canon/snapshots/results/taylors-power-law.json
    canon/snapshots/figures/taylors-power-law.png
    canon/patterns/taylors-power-law.md  (checks[0] updated in-place)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Optional matplotlib — needed for the figure; not needed for offline tests.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

CANON_ROOT = Path(__file__).parent.parent
SNAPSHOT_DIR = CANON_ROOT / "snapshots" / "neon"
RESULTS_DIR = CANON_ROOT / "snapshots" / "results"
FIGURES_DIR = CANON_ROOT / "snapshots" / "figures"
NODE_PATH = CANON_ROOT / "patterns" / "taylors-power-law.md"


# ---------------------------------------------------------------------------
# Data aggregation
# ---------------------------------------------------------------------------

def aggregate_counts(rows: list[dict]) -> dict[tuple[str, str, str], float]:
    """Sum clusterSize by (siteID, pointID, taxonID).

    Returns a dict mapping (site, point, taxon) → total count.
    """
    agg: dict[tuple[str, str, str], float] = defaultdict(float)
    for row in rows:
        site = (row.get("siteID") or "").strip()
        point = (row.get("pointID") or "").strip()
        taxon = (row.get("taxonID") or "").strip()
        cluster = row.get("clusterSize") or row.get("observerDistance") or ""
        if not (site and point and taxon):
            continue
        try:
            count = float(cluster) if cluster else 1.0
        except (ValueError, TypeError):
            count = 1.0
        agg[(site, point, taxon)] += count
    return dict(agg)


def fill_zeros(agg: dict[tuple[str, str, str], float]) -> dict[tuple[str, str, str], float]:
    """Add zero entries for (site, point, taxon) where taxon seen at site but not point.

    For each site, any taxon detected anywhere at that site gets a 0 count
    at every point where it was not explicitly observed.
    """
    # Group by site
    sites: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"points": set(), "taxa": set()})
    for (site, point, taxon) in agg:
        sites[site]["points"].add(point)
        sites[site]["taxa"].add(taxon)

    result = dict(agg)
    for site, info in sites.items():
        for taxon in info["taxa"]:
            for point in info["points"]:
                key = (site, point, taxon)
                if key not in result:
                    result[key] = 0.0
    return result


def compute_means_and_variances(
    agg: dict[tuple[str, str, str], float],
    min_samples: int = 20,
) -> list[tuple[float, float]]:
    """Return (mean, variance) pairs per taxon with ≥ min_samples (site, point) observations.

    Groups all (site, point) samples by taxonID.
    """
    by_taxon: dict[str, list[float]] = defaultdict(list)
    for (_, _, taxon), count in agg.items():
        by_taxon[taxon].append(count)

    pairs = []
    for taxon, counts in by_taxon.items():
        if len(counts) < min_samples:
            continue
        arr = np.array(counts, dtype=float)
        m = float(np.mean(arr))
        v = float(np.var(arr, ddof=1))
        if m > 0 and v > 0:
            pairs.append((m, v))
    return pairs


def compute_site_means_and_variances(
    agg: dict[tuple[str, str, str], float],
    site: str,
    min_samples: int = 5,
) -> list[tuple[float, float]]:
    """Return (mean, variance) pairs for one site."""
    by_taxon: dict[str, list[float]] = defaultdict(list)
    for (s, _, taxon), count in agg.items():
        if s == site:
            by_taxon[taxon].append(count)

    pairs = []
    for _, counts in by_taxon.items():
        if len(counts) < min_samples:
            continue
        arr = np.array(counts, dtype=float)
        m = float(np.mean(arr))
        v = float(np.var(arr, ddof=1))
        if m > 0 and v > 0:
            pairs.append((m, v))
    return pairs


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _ols_log_fit(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """OLS fit of log10(var) = log10(a) + b * log10(mean). Returns (a, b)."""
    X = np.log10([p[0] for p in pairs])
    Y = np.log10([p[1] for p in pairs])
    A = np.column_stack([np.ones_like(X), X])
    coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    intercept, b = float(coeffs[0]), float(coeffs[1])
    return 10 ** intercept, b


def fit_tpl(
    pairs: list[tuple[float, float]],
    n_bootstrap: int = 1000,
    rng_seed: Optional[int] = 42,
) -> dict:
    """Fit Taylor's power law by OLS in log-log space with bootstrap CI.

    Args:
        pairs: List of (mean, variance) pairs, one per species.
        n_bootstrap: Number of bootstrap resamples for 95% CI on b.
        rng_seed: Random seed for reproducibility.

    Returns:
        dict with keys: a, b, b_ci95 ([lo, hi]), n_species, pass.
    """
    if len(pairs) < 2:
        return {"a": None, "b": None, "b_ci95": None, "n_species": len(pairs), "pass": False}

    a, b = _ols_log_fit(pairs)

    # Bootstrap CI on b
    rng = np.random.default_rng(rng_seed)
    b_samples = []
    n = len(pairs)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = [pairs[int(i)] for i in idx]
        if len(set(sample)) < 2:
            continue
        try:
            _, b_s = _ols_log_fit(sample)
            b_samples.append(b_s)
        except (np.linalg.LinAlgError, ValueError):
            continue

    if len(b_samples) >= 10:
        b_ci95 = np.percentile(b_samples, [2.5, 97.5]).tolist()
    else:
        b_ci95 = [b, b]

    return {
        "a": round(a, 4),
        "b": round(b, 4),
        "b_ci95": [round(b_ci95[0], 4), round(b_ci95[1], 4)],
        "n_species": n,
        "pass": 1.0 <= b <= 2.0,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def save_figure(
    pairs: list[tuple[float, float]],
    fit: dict,
    output_path: Path,
) -> None:
    """Log-log scatter of var vs mean with fitted line."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available — skipping figure", file=sys.stderr)
        return

    means = np.array([p[0] for p in pairs])
    variances = np.array([p[1] for p in pairs])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(means, variances, s=18, alpha=0.6, color="#2563eb", label="Species")

    if fit["a"] is not None:
        x_range = np.logspace(np.log10(means.min()), np.log10(means.max()), 200)
        y_fit = fit["a"] * x_range ** fit["b"]
        ax.plot(x_range, y_fit, color="#dc2626", linewidth=1.5,
                label=f"b = {fit['b']:.3f} (95% CI [{fit['b_ci95'][0]:.3f}, {fit['b_ci95'][1]:.3f}])")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean count per (site, point)")
    ax.set_ylabel("Variance of count")
    ax.set_title(
        f"Taylor's power law — NEON landbird counts\n"
        f"b = {fit['b']:.3f}, 95% CI [{fit['b_ci95'][0]:.3f}, {fit['b_ci95'][1]:.3f}], "
        f"n = {fit['n_species']} species"
    )
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Node file update
# ---------------------------------------------------------------------------

def update_node_check(node_path: Path, check_id: str, result: dict, last_run: str) -> None:
    """Update checks[i].last_run and checks[i].result in a canon node file."""
    try:
        import frontmatter
    except ImportError:
        print("python-frontmatter not available — skipping node update", file=sys.stderr)
        return

    post = frontmatter.load(str(node_path))
    updated = False
    for check in post.metadata.get("checks", []):
        if check.get("id") == check_id:
            check["last_run"] = last_run
            check["result"] = result
            updated = True
            break
    if updated:
        with open(node_path, "wb") as f:
            frontmatter.dump(post, f)


# ---------------------------------------------------------------------------
# NEON download
# ---------------------------------------------------------------------------

async def _fetch_site_month(site: str, year_month: str, cache_dir: Path) -> list[dict]:
    """Download brd_countdata CSV for one site-month, using cache if present."""
    import httpx

    cached_glob = list(cache_dir.glob(f"NEON.*.{site}.DP1.10003.001.brd_countdata.{year_month}.basic.*.csv"))
    if cached_glob:
        import csv as _csv
        import io as _io
        rows = []
        for path in cached_glob:
            text = path.read_text(encoding="utf-8")
            rows.extend(_csv.DictReader(_io.StringIO(text)))
        return rows

    url = f"https://data.neonscience.org/api/v0/data/DP1.10003.001/{site}/{year_month}"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            file_list = resp.json().get("data", {}).get("files", [])
    except Exception as exc:
        print(f"  {site} {year_month}: download failed ({exc})", file=sys.stderr)
        return []

    import csv as _csv
    import io as _io
    rows = []
    for file_info in file_list:
        name = file_info.get("name", "")
        if not name.endswith(".csv") or "brd_countdata" not in name or "expanded" in name:
            continue
        file_url = file_info.get("url", "")
        if not file_url:
            continue
        time.sleep(0.5)
        try:
            async with httpx.AsyncClient(timeout=120) as dl:
                r = await dl.get(file_url)
                r.raise_for_status()
                csv_text = r.text
        except Exception as exc:
            print(f"  {site} {year_month} {name}: file download failed ({exc})", file=sys.stderr)
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / name).write_text(csv_text, encoding="utf-8")
        rows.extend(_csv.DictReader(_io.StringIO(csv_text)))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_check(sites: list[str], year: int) -> dict:
    """Download NEON data, fit Taylor's power law, write results and figure."""
    print(f"Fetching NEON DP1.10003.001 for {len(sites)} sites, year {year}...")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for site in sites:
        for month in [f"{year}-06", f"{year}-07"]:
            print(f"  {site} {month}...", end=" ", flush=True)
            rows = await _fetch_site_month(site, month, SNAPSHOT_DIR)
            all_rows.extend(rows)
            print(f"{len(rows)} rows")

    print(f"Total rows: {len(all_rows)}")
    if not all_rows:
        print("No data — check network access and site codes.", file=sys.stderr)
        sys.exit(1)

    agg = aggregate_counts(all_rows)
    agg = fill_zeros(agg)
    pairs = compute_means_and_variances(agg, min_samples=20)
    print(f"Species qualifying for global fit: {len(pairs)}")

    fit = fit_tpl(pairs)
    print(f"Global fit: b = {fit['b']}, 95% CI {fit['b_ci95']}, pass = {fit['pass']}")

    # Per-site fits
    site_set = {s for (s, _, _) in agg}
    per_site: dict[str, dict] = {}
    for site in site_set:
        site_pairs = compute_site_means_and_variances(agg, site)
        site_fit = fit_tpl(site_pairs)
        if site_fit["n_species"] >= 15:
            per_site[site] = {"b": site_fit["b"], "n_species": site_fit["n_species"]}

    result = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "sites": sites,
        "year": year,
        "n_species": fit["n_species"],
        "b": fit["b"],
        "b_ci95": fit["b_ci95"],
        "a": fit["a"],
        "per_site": per_site,
        "pass": fit["pass"],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "taylors-power-law.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Results written to {result_path}")

    figure_path = FIGURES_DIR / "taylors-power-law.png"
    save_figure(pairs, fit, figure_path)
    if HAS_MATPLOTLIB:
        print(f"Figure written to {figure_path}")

    if NODE_PATH.exists():
        update_node_check(
            NODE_PATH,
            check_id="neon-landbird-tpl",
            result=result,
            last_run=result["run_at"],
        )
        print(f"Node file updated: {NODE_PATH}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Taylor's power law on NEON landbird counts.")
    parser.add_argument(
        "--sites",
        default="WREF,HARV,SRER,OSBS,KONZ,CPER,TALL,BART,UNDE,JERC",
        help="Comma-separated NEON site codes.",
    )
    parser.add_argument("--year", type=int, default=2023, help="Survey year.")
    args = parser.parse_args()
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    asyncio.run(run_check(sites, args.year))


if __name__ == "__main__":
    main()
