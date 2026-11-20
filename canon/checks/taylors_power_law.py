"""Check: Taylor's power law on NEON breeding-landbird point counts (spec 018).

    uv run --package kinship-orchestrator python canon/checks/taylors_power_law.py \
        --sites WREF,HARV,SRER,OSBS,KONZ,CPER,TALL,BART,UNDE,JERC --year 2023

Var(N) = a * mean(N)^b. For each species, counts per (site, point) over the
breeding season give one mean and one variance; b is the OLS slope of
log10(var) on log10(mean) across species, with a bootstrap 95% CI.

Uses true counts only. Presence-only sources (GBIF, iNaturalist) cannot test
this law and must never be fed to it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NODE = ROOT / "canon" / "patterns" / "taylors-power-law.md"
CACHE = ROOT / "canon" / "snapshots" / "neon"
RESULT = ROOT / "canon" / "snapshots" / "results" / "taylors-power-law.json"
FIGURE = ROOT / "canon" / "snapshots" / "figures" / "taylors-power-law.png"
PRODUCT = "DP1.10003.001"
TABLE = "countdata"
BREEDING_MONTHS = {"04", "05", "06", "07"}  # NEON surveys one peak-season window per site


def aggregate(rows: list[dict]) -> dict[str, dict[tuple[str, str], float]]:
    """Counts per species per (site, point) for one season.

    Sums clusterSize over every record at the point. A point surveyed at a
    site where the species was detected, but with no record of it, counts
    as 0: zeros matter for the variance.
    """
    counts: dict[str, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    points_by_site: dict[str, set[str]] = defaultdict(set)
    species_by_site: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        site, point, taxon = r.get("siteID"), r.get("pointID"), r.get("taxonID")
        if not site or not point:
            continue
        points_by_site[site].add(point)
        if not taxon:
            continue
        try:
            size = float(r.get("clusterSize") or 0)
        except ValueError:
            continue
        counts[taxon][(site, point)] += size
        species_by_site[site].add(taxon)
    for site, species in species_by_site.items():
        for taxon in species:
            for point in points_by_site[site]:
                counts[taxon].setdefault((site, point), 0.0)
    return {t: dict(v) for t, v in counts.items()}


def mean_var(counts: dict[str, dict[tuple[str, str], float]], min_samples: int = 20,
             site: str | None = None) -> dict[str, tuple[float, float, int]]:
    """Per species (mean, sample variance, n) over its samples, optionally one site."""
    out = {}
    for taxon, by_sample in counts.items():
        vals = [v for (s, _), v in by_sample.items() if site is None or s == site]
        if len(vals) >= min_samples:
            arr = np.asarray(vals, dtype=float)
            out[taxon] = (float(arr.mean()), float(arr.var(ddof=1)), len(vals))
    return out


def fit_power_law(means, variances) -> tuple[float, float, int]:
    """OLS of log10(var) on log10(mean). Returns (a, b, n_used); positive pairs only."""
    m = np.asarray(means, dtype=float)
    v = np.asarray(variances, dtype=float)
    keep = (m > 0) & (v > 0)
    x, y = np.log10(m[keep]), np.log10(v[keep])
    if len(x) < 3:
        raise ValueError("need at least 3 species with positive mean and variance")
    b, log_a = np.polyfit(x, y, 1)
    return float(10 ** log_a), float(b), int(keep.sum())


def bootstrap_b(means, variances, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = np.asarray(means, dtype=float)
    v = np.asarray(variances, dtype=float)
    keep = (m > 0) & (v > 0)
    m, v = m[keep], v[keep]
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(m), len(m))
        if len(set(idx.tolist())) < 3:
            continue
        bs.append(np.polyfit(np.log10(m[idx]), np.log10(v[idx]), 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


async def download(sites: list[str], year: int) -> list[dict]:
    from neonscience_mcp.adapter import NeonAdapter
    adapter = NeonAdapter()
    product = await adapter.get_data_product(PRODUCT) or {}
    months = {s["siteCode"]: s.get("availableMonths", []) for s in product.get("siteCodes", [])}
    rows: list[dict] = []
    for site in sites:
        for ym in months.get(site, []):
            if ym.startswith(f"{year}-") and ym[5:7] in BREEDING_MONTHS:
                rows += await adapter.get_product_data(PRODUCT, site, ym, TABLE, cache_dir=CACHE)
                await asyncio.sleep(0.5)
    return rows


def cached_rows(sites: list[str], year: int) -> list[dict]:
    import csv
    rows = []
    for path in sorted(CACHE.glob(f"{PRODUCT}_*_{year}-*_{TABLE}.csv")):
        if path.name.split("_")[1] in sites:
            rows += list(csv.DictReader(path.open()))
    return rows


def run(rows: list[dict], sites: list[str], year: int) -> dict:
    counts = aggregate(rows)
    mv = mean_var(counts, min_samples=20)
    taxa = sorted(mv)
    means = [mv[t][0] for t in taxa]
    variances = [mv[t][1] for t in taxa]
    a, b, n = fit_power_law(means, variances)
    lo, hi = bootstrap_b(means, variances)
    per_site = {}
    for site in sites:
        smv = mean_var(counts, min_samples=20, site=site)
        if len(smv) >= 15:
            try:
                _, sb, sn = fit_power_law([x[0] for x in smv.values()], [x[1] for x in smv.values()])
                per_site[site] = {"b": round(sb, 3), "n_species": sn}
            except ValueError:
                pass
    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sites": sites, "year": year, "product": PRODUCT,
        "n_species": n, "b": round(b, 3), "b_ci95": [round(lo, 3), round(hi, 3)],
        "a": round(a, 4), "per_site": per_site,
        "pass": bool(1.0 <= b <= 2.0),
        "_points": {t: [round(mv[t][0], 4), round(mv[t][1], 4)] for t in taxa},
    }


def write_figure(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = np.asarray(list(result["_points"].values()))
    keep = (pts[:, 0] > 0) & (pts[:, 1] > 0)
    pts = pts[keep]
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=130)
    ax.loglog(pts[:, 0], pts[:, 1], "o", ms=4, alpha=0.7, color="#3b6e8f")
    xs = np.logspace(np.log10(pts[:, 0].min()), np.log10(pts[:, 0].max()), 50)
    ax.loglog(xs, result["a"] * xs ** result["b"], color="#b5482f", lw=1.5)
    ax.set_xlabel("Mean count per point")
    ax.set_ylabel("Variance of count")
    lo, hi = result["b_ci95"]
    ax.set_title(f"Taylor's power law, NEON landbirds {result['year']}\n"
                 f"b = {result['b']:.2f} (95% CI {lo:.2f}–{hi:.2f}), {result['n_species']} species",
                 fontsize=9)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE)
    plt.close(fig)


def update_node(result: dict) -> None:
    import frontmatter
    post = frontmatter.load(NODE)
    checks = post.metadata.get("checks") or []
    if checks:
        checks[0]["last_run"] = result["run_at"]
        checks[0]["result"] = {
            "b": result["b"], "b_ci95": result["b_ci95"], "n_species": result["n_species"],
            "pass": result["pass"], "snapshot": "canon/snapshots/results/taylors-power-law.json",
        }
    NODE.write_text(frontmatter.dumps(post) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="WREF,HARV,SRER,OSBS,KONZ,CPER,TALL,BART,UNDE,JERC")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--offline", action="store_true", help="use cached CSVs only")
    args = ap.parse_args()
    sites = [s.strip().upper() for s in args.sites.split(",") if s.strip()]
    rows = cached_rows(sites, args.year) if args.offline else asyncio.run(download(sites, args.year))
    if not rows:
        raise SystemExit("No NEON rows (no cache, or the data endpoint refused; set NEON_API_TOKEN)")
    result = run(rows, sites, args.year)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    write_figure(result)
    update_node(result)
    print(f"b = {result['b']} (95% CI {result['b_ci95']}), {result['n_species']} species, "
          f"pass = {result['pass']}")


if __name__ == "__main__":
    main()
