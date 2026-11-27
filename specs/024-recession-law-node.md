# Spec 024: Second Canon Node — Streamflow Recession Law on USGS Gauges

> Phase 6 — The Canon, rivers first (docs/DIRECTION.md)
> Priority: P1
> Estimated effort: 1 session
> Dependency: Specs 015 (`get_daily_record`), 017 (schema), 018 (check-runner pattern), 021 (canon CI)
> Network: `api.waterdata.usgs.gov` only, which the runner reaches today.
> Written 2026-11-27 because every other spec is blocked on hosts or a token.

## Objective

Add the first **river** pattern to the canon, with a check that runs on data
the federation already fetches and caches: how fast a river falls after rain
stops. It is one of the oldest results in hydrology, it ties streamflow to
the groundwater that feeds it, and its parameters **drift** when aquifers are
drawn down or droughts deepen. That makes it the first node where the canon
can track a pattern changing over time, as DIRECTION.md asks.

## The pattern

During dry spells, flow Q falls as **−dQ/dt = a · Q^b**. On a log–log plot of
the daily fall against flow, recession days lie near a straight line with
slope *b*, commonly reported between about 1 and 2 for whole catchments.
`a` sets the speed: larger means the river drains faster.

Origin: Brutsaert and Nieber (1977), Water Resources Research.
**Confirm the full citation and DOI from an opened page before committing
the node** (the citation is written from memory here; the canon forbids
that in committed files). If it cannot be opened, cite it without a DOI and
add a note.

## What to Build

### 1. Recession extraction and fit — `canon/checks/recession_law.py`

```
uv run --package kinship-orchestrator python canon/checks/recession_law.py \
    --sites USGS-11467000,USGS-11477000,USGS-11425500,USGS-11303500,USGS-11530500 \
    [--offline]
```

1. Load each gauge's full daily record with `USGSNWISAdapter.get_daily_record`
   (cached; `--offline` reads `~/.kinship-earth/cache/usgs-daily/` or a copy
   committed under `canon/snapshots/usgs/` — commit a copy of only the
   gauges used, since the cache lives outside the repo).
2. Recession days: runs of ≥ 5 consecutive days of falling flow
   (Q[t+1] < Q[t]); drop the first 2 days of each run (quickflow) and any
   day with Q ≤ 0. Use the central difference −dQ/dt ≈ Q[t] − Q[t+1]
   against the mean of the two days.
3. Fit log10(−dQ/dt) = log10(a) + b · log10(Q) per gauge by OLS on all
   recession days; bootstrap the **recession events** (not days) 1,000
   times for a 95% CI on *b*. numpy only.
4. **Drift:** repeat the fit per decade (≥ 200 recession days per decade)
   and report *b* and *a* by decade, plus the OLS slope of log10(a) over
   decade midpoints with a CI. That is the "how the pattern is changing"
   number.
5. Result JSON `canon/snapshots/results/recession-law.json`:
   `{run_at, sites, per_gauge: {site: {b, b_ci95, a, n_days, n_events, by_decade: [...]}}, pass}`,
   with `pass = true` when *b* lies in [1.0, 2.5] at a majority of gauges.
   The range is the stated expectation, written before the run; do not
   widen it after seeing results.
6. Figure `canon/snapshots/figures/recession-law.png`: one log–log panel
   per gauge (recession points faint, fitted line, *b* with CI in the
   title), plus one small panel of *b* by decade per gauge.
7. Update the node's `checks[0].last_run` / `result` the same way
   `taylors_power_law.py` does (reuse its `update_node` pattern).

### 2. Node file — `canon/patterns/streamflow-recession-law.md`

- `kind: empirical_law`, `status: conjecture` (a human reviewer decides
  `supported`), `layer: ecosystem`.
- `regime.conditions`: "recession periods only (no rain, no snowmelt
  input)", "unregulated or lightly regulated rivers; dam releases break
  it". Note that 4 of the 5 suggested gauges carry USGS "regulated" flags,
  and add one unregulated gauge if one with ≥ 30 years can be found via
  `time-series-metadata` (say which and why in the body).
- `known_exceptions`: snowmelt recessions; regulated reaches; *b* depends on
  the fitting method (state this plainly; it is the main critique in the
  literature).
- Edges: `INFORMS` → `baseflow-groundwater-storage` (stub);
  `ANALOG_OF` → `nonlinear-reservoir-drainage` (stub, field: physics,
  note: "storage–discharge as a nonlinear reservoir"); `INFORMS` →
  `taylors-power-law` (both are power laws whose exponent is argued to be
  partly a fitting artefact; mark `confidence: conjectural`).
- Body: plain language — what it says, why hydrologists use it, where it
  breaks, and what drift in *a* would mean for a rancher or a fire district
  (rivers dropping faster in late summer).

### 3. Expose drift in the tools

`canon_check_status` already returns the stored result; make sure the
by-decade list survives into it (no code change expected; add a test).

## Tests — `canon/checks/tests/test_recession_law.py`

1. Synthetic recession generated from −dQ/dt = a·Q^b with known a, b and
   noise: the fitter recovers *b* within 0.1.
2. Recession extraction: a hand-built series with a rain spike drops the
   rising days and the first 2 falling days, and keeps runs of ≥ 5 only.
3. Decade drift: synthetic record whose *a* doubles in the last decade gives
   a positive drift slope whose CI excludes 0.
4. Offline snapshot run (`@pytest.mark.network`-free if the snapshot is
   committed): completes and writes the result JSON.
5. `tools/canon_ci.py canon/patterns/` passes with the new node and stubs.

## Do Not

- Do not tune the expected *b* range after seeing results.
- Do not mark the node `supported`.
- Do not include snowmelt-season days at the Sierra gauge without saying so;
  prefer to restrict that gauge to August–October recessions and note it.

## Commit

```
Add canon node: streamflow recession law on USGS gauges (spec 024)

- canon/checks/recession_law.py: recession extraction, fit, event bootstrap,
  per-decade drift, figure
- canon/patterns/streamflow-recession-law.md (status: conjecture)
- snapshot + figure; 5 tests
```
