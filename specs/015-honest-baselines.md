# Spec 015: Honest Baselines

> Phase 6 — "Is this normal for here?" (see docs/DIRECTION.md)
> Priority: P0
> Estimated effort: 1 session
> Dependency: None

## Objective

Make "normal" mean "this place's own long record for this time of year." Today streamflow normal is the last 30 days pooled across neighbouring gauges, climate normal is the last 5 years, and flow anomalies use percent-of-mean on a skewed variable. Fix all three without breaking existing callers.

## Background (what the code does now)

- `shared/src/kinship_shared/baselines.py`
  - `compute_baselines_from_usgs(usgs_adapter, lat, lng, target_date, radius_km=50)` searches `target_date − 30 days … target_date`, pools every observation from up to 5 nearby gauges, returns `streamflow_mean_cfs` / `streamflow_std_cfs`.
  - `compute_baselines_from_era5(..., years_back=5)` averages a ±7-day window around the day-of-year over 5 prior years.
- `shared/src/kinship_shared/state_builder.py` calls both (`baseline_years: int = 5`) and writes `streamflow_baseline` onto `EcosystemState`.
- `shared/src/kinship_shared/anomaly_detect.py` `FlowDetector` fires when flow is ±50% from `state.streamflow_baseline`.
- `servers/usgs-nwis/src/*/adapter.py` uses `https://api.waterdata.usgs.gov/ogcapi/v0`, `collections/daily/items` with `statistic_id=00003` for date ranges, capped at `limit ≤ 100` and 5 sites.

## What to Build

### 1. USGS adapter: long daily record for one gauge

Add to `USGSNWISAdapter`:

```python
async def get_daily_record(
    self, site_id: str, parameter_code: str = "00060",
    start_date: str | None = None, end_date: str | None = None,
) -> list[tuple[date, float]]:
    """Full daily-mean record for one monitoring location, oldest first."""
```

- Page through `collections/daily/items` until exhausted.
- **Before building, make one live call and confirm:** how pagination works on this API (next link vs. offset), the maximum page size, and how far back the daily record goes for a long-running gauge (try `USGS-11467000`, Russian River near Guerneville). Do not assume; record what you find in a comment at the top of the method. The adapter's `temporal_coverage_start="2007-10-01"` describes the continuous (15-minute) data and may not apply to daily values; correct the capability text if daily values go back further.
- Respect the existing rate limit (16/min) and use `http_get_with_retry`.
- Cache the result on disk under `~/.kinship-earth/cache/usgs-daily/{site_id}_{parameter}.json` with the fetch date; reuse if under 7 days old.

### 2. Day-of-year flow climatology — `baselines.py`

```python
class FlowClimatology(BaseModel):
    site_id: str
    day_of_year: int
    window_days: int            # half-width, default 7
    years_of_data: int
    n_values: int
    p05: float; p10: float; p25: float; p50: float; p75: float; p90: float; p95: float
    record_start: date
    record_end: date

def flow_climatology(record: list[tuple[date, float]], target: date, window_days: int = 7,
                     min_years: int = 10) -> FlowClimatology | None
def flow_percentile(value: float, record: list[tuple[date, float]], target: date,
                    window_days: int = 7) -> float | None   # 0–100, empirical
```

- Use all values within ±`window_days` of the target day-of-year across all years, excluding the target year itself.
- Return `None` if fewer than `min_years` distinct years contribute. Never fall back silently to a short window.
- Pure functions, numpy optional (stdlib is fine).

### 3. Replace the pooled 30-day baseline

Rewrite `compute_baselines_from_usgs` to:

1. Find the nearest gauges as now.
2. Pick **one** gauge (nearest with a daily discharge record ≥ `min_years`); do not pool gauges.
3. Return a dict with the existing keys kept for compatibility — `streamflow_mean_cfs` now set to the day-of-year **median (p50)**, `streamflow_std_cfs` to `(p75 − p25) / 1.349` — plus new keys: `site_id`, `streamflow_percentile` (of the current value when supplied), `climatology` (the `FlowClimatology` as a dict), `baseline_kind: "doy_record"`.
4. If no gauge has enough record, return `{"baseline_kind": "insufficient_record"}` and no streamflow keys. Callers must then skip flow anomaly detection.

Add an optional `current_flow: float | None = None` parameter for step 3's percentile.

### 4. Thirty-year climate normals

- `compute_baselines_from_era5`: change default `years_back` to 30. Fetch efficiently: one `get_daily` call per year is 30 calls; if the adapter (Open-Meteo archive) accepts a multi-year range, fetch once and slice by day-of-year instead. Check and use whichever is correct.
- Also compute and return on `BaselineValues` (new optional fields): `temp_trend_c_per_decade` (OLS slope of the yearly window means) and `baseline_period` (e.g. `"1996–2025"`).
- `state_builder.build_ecosystem_state`: change `baseline_years` default to 30.

### 5. Percentile-based flow anomalies

Extend `EcosystemState` with optional `streamflow_percentile: float | None` and `streamflow_baseline_kind: str | None`; populate both in `state_builder`.

`FlowDetector.detect`:

- If `state.streamflow_percentile` is present: fire when percentile ≤ 10 (low) or ≥ 90 (high). Severity: ≤ 2 or ≥ 98 → `critical`; ≤ 5 or ≥ 95 → `warning`; else `info`. Description must state the percentile and the record length, e.g. "Streamflow at the 4th percentile for this date (62 years of record, USGS-11467000): 84 cfs vs median 310 cfs."
- If absent, keep the existing percent-of-mean path unchanged so current tests pass.
- Keep `deviation_pct` populated (percent from median) for downstream narrative templates.

## Tests

`shared/tests/test_flow_climatology.py` (offline, synthetic records):

1. A 40-year synthetic record with a known seasonal cycle: `flow_climatology` returns p50 within 5% of the constructed median for a given day.
2. Year-end wrap: a target of Jan 3 includes late-December values.
3. Target year excluded from its own baseline.
4. Fewer than `min_years` → `None`.
5. `flow_percentile` returns ~5 for a value at the constructed 5th percentile, and 0 / 100 at the extremes.

`shared/tests/test_anomaly_detect.py` (add):

6. State with `streamflow_percentile=3` → one `warning` flow anomaly whose description contains "percentile".
7. State with `streamflow_percentile=50` → none.
8. State without percentile → existing behaviour (existing tests untouched and passing).

Network test (mark `@pytest.mark.network`, skipped offline):

9. `get_daily_record("USGS-11467000")` returns more than 10 years of values in date order.

All existing tests in `shared/tests/` and `servers/orchestrator/tests/` must still pass.

## Do Not

- Do not pool multiple gauges into one baseline.
- Do not change `EcologicalAnomaly` required fields.
- Do not add new MCP tools.
- Do not hard-code a site list.

## Definition of Done

- `ecology_ecosystem_state` and `ecology_check_anomalies` report flow against the gauge's own day-of-year record, naming the gauge and record length.
- Climate normals are 30-year with a reported trend.
- README's monitoring section gains two sentences on what "normal" means.

## Commit

```
Honest baselines: per-gauge day-of-year flow record, 30-year climate normals (spec 015)

- USGS adapter: get_daily_record with paging and disk cache
- baselines: FlowClimatology, flow_percentile; single-gauge baseline, no pooling
- ERA5 normals default to 30 years, with trend per decade
- FlowDetector: percentile thresholds when available
- 9 tests
```
