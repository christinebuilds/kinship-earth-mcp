# Spec 018: First Canon Node — Taylor's Power Law on NEON Landbird Counts

> Phase 6 — The Canon (see docs/CANON.md)
> Priority: P0
> Estimated effort: 1 session
> Dependency: Spec 017

## Objective

Ship the first real canon node end to end: node file, executable check against NEON breeding-landbird point counts (DP1.10003.001), committed result snapshot, one figure. This proves the pipeline. If the fitted exponent lands in [1, 2] the pipeline works; if it does not, the bug is almost certainly in the data handling, because the law is among the best-attested in ecology.

## Background

Taylor's power law (Taylor 1961, Nature 189:732, doi:10.1038/189732a0): for repeated counts of a species, `Var(N) = a · mean(N)^b`, with `b` typically between 1 and 2. Eisler, Bartos & Kertész 2008 (Adv. Phys. 57:89, doi:10.1080/00018730801893043) review the same form across physics, finance and ecology ("fluctuation scaling"), which is the node's cross-field edge.

## What to Build

### 1. NEON tabular data download — extend `servers/neonscience/src/neonscience_mcp/adapter.py`

The adapter currently lists sites and products but does not fetch product data files. Add:

```python
async def get_product_data(self, product_code: str, site_code: str, year_month: str) -> list[dict]:
    """Download and parse the basic-package CSV tables for one product/site/month.
    Uses GET https://data.neonscience.org/api/v0/data/{product}/{site}/{year_month}
    and returns rows from the table whose name contains the given substring."""
```

Add a `table` parameter (default `"countdata"` for this product). Parse CSV with the stdlib `csv` module. Cache downloaded files under `canon/snapshots/neon/` so re-runs are offline. Respect NEON's rate guidance (sequential requests, 0.5 s sleep).

### 2. Check runner — `canon/checks/taylors_power_law.py`

```
uv run python canon/checks/taylors_power_law.py --sites WREF,HARV,SRER,OSBS,KONZ,CPER,TALL,BART,UNDE,JERC --year 2023
```

Steps:
1. For each site, download `DP1.10003.001` for June and July of the year (breeding season). Rows of `brd_countdata` give `taxonID`, `clusterSize`, `pointID`, `startDate`.
2. Aggregate to counts per (site, pointID, taxonID) = sum of `clusterSize` over the visit. Points with no detection of a species count as 0 for that species if the species was detected anywhere at that site that season (zeros matter for the variance).
3. For each species with ≥ 20 (site, point) samples: compute mean and variance across samples.
4. Fit `log10(var) = log10(a) + b · log10(mean)` by OLS over species with mean > 0, var > 0. Bootstrap 1,000 resamples for a 95% CI on `b`.
5. Also fit per-site `b` where a site has ≥ 15 qualifying species.
6. Write `canon/snapshots/results/taylors-power-law.json`:
   `{run_at, sites, year, n_species, b, b_ci95, a, per_site: {site: {b, n_species}}, pass: bool}` with `pass = 1.0 <= b <= 2.0`.
7. Save figure `canon/snapshots/figures/taylors-power-law.png`: log–log scatter of var vs mean, one point per species, fitted line, `b` and CI in the title. Matplotlib, no seaborn.
8. Update the node file's `checks[0].last_run` and `checks[0].result` (use `canon_loader` to round-trip the front matter).

Use numpy only (no scipy/statsmodels dependency).

### 3. Node file — `canon/patterns/taylors-power-law.md`

Front matter exactly as drafted in docs/CANON.md ("The object: one node"), with the `checks[0]` block pointing at the runner above. Body (~300 words, plain language):
- What it says, with the formula in words.
- Why it is believed: observed in hundreds of species since 1961; the Eisler review.
- Where it breaks: `b` outside [1, 2] for strongly aggregated populations; presence-only data cannot test it (state this explicitly so nobody runs it on GBIF).
- What it connects to: Tweedie distributions, fluctuation scaling in physics and finance, density dependence.
- The open question: whether `b` is biological or statistical (cite Kendal 2004 and Cohen & Xu 2015, PNAS 112:7749, doi:10.1073/pnas.1503824112 — the latter shows `b` can arise from sampling alone).

Add stubs for `tweedie-distributions`, `fluctuation-scaling-physics`, `density-dependence` to `canon/patterns/_stubs.yaml`.

### 4. Validate and load

- `tools/canon_validate.py` passes.
- `canon_to_graph` loads the node into the SQLite graph store; `ecology_memory_recall("Taylor's law")` returns the pattern entity (sanity check that the existing memory tools see canon nodes).

## Tests — `canon/checks/tests/test_taylors_power_law.py`

1. Synthetic data test: generate Poisson counts (true `b = 1`) and negative-binomial counts (true `b ≈ 1.5`); the fitter recovers each within 0.1. Offline.
2. Aggregation test: a hand-built table of 3 points × 2 species, including a zero, produces the expected means and variances. Offline.
3. Snapshot test: if `canon/snapshots/neon/` contains cached files, the full run completes and `pass` is true. Skipped if no cache (network test, marked `@pytest.mark.network`).

## Do Not

- Do not fit on presence-only sources (GBIF, iNaturalist). Record this prohibition in the node.
- Do not add new MCP tools (spec 020).
- Do not add eBird as a second check yet; note it as a TODO in the node.

## Definition of Done

- Node validates, loads into the graph, has a passing check result and a committed figure.
- `b` and its CI are in the node file and the snapshot JSON.
- README gets one paragraph: "Kinship Earth now carries a canon of ecological patterns; here is the first one" with the figure.

## Commit

```
Add first canon node: Taylor's power law on NEON landbird counts (spec 018)

- NEON adapter: get_product_data for tabular products
- canon/checks/taylors_power_law.py with bootstrap CI and figure
- canon/patterns/taylors-power-law.md (status: supported, b=<value>)
- snapshot + figure committed; 3 tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
```
