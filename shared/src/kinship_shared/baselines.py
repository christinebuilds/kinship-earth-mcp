"""
Baseline computation for ecosystem monitoring.

Computes historical normals from ERA5 (climate) and USGS NWIS (hydrology)
to establish what 'normal' looks like for a location at a given time of year.
Baselines are the reference against which anomalies are detected.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Optional

from pydantic import BaseModel, Field


class BaselineValues(BaseModel):
    """Historical normal values for a location at a specific day-of-year."""

    location_id: str = Field(description="e.g. 'location:41.50_-70.70'")
    day_of_year: int = Field(ge=1, le=366)
    years_of_data: int = Field(default=0)

    temp_mean_c: Optional[float] = None
    temp_std_c: Optional[float] = None
    precip_mean_mm: Optional[float] = None
    precip_std_mm: Optional[float] = None
    temp_trend_c_per_decade: Optional[float] = None
    baseline_period: Optional[str] = None

    streamflow_mean_cfs: Optional[float] = None
    streamflow_std_cfs: Optional[float] = None

    species_richness_mean: Optional[int] = None
    ndvi_mean: Optional[float] = None
    ndvi_std: Optional[float] = None


class FlowClimatology(BaseModel):
    """Day-of-year flow statistics built from a site's full daily record."""

    site_id: str
    day_of_year: int
    window_days: int
    years_of_data: int
    n_values: int
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    record_start: date
    record_end: date


def flow_climatology(
    record: list[tuple[date, float]],
    target: date,
    window_days: int = 7,
    min_years: int = 10,
    *,
    site_id: str = "",
) -> FlowClimatology | None:
    """Build day-of-year flow statistics from a full daily record.

    Collects all values within ±window_days of the target's day-of-year
    across all years, excluding the target year itself.

    Returns None when fewer than min_years distinct years contribute.
    """
    if not record:
        return None

    target_doy = target.timetuple().tm_yday
    target_year = target.year

    values: list[float] = []
    years_seen: set[int] = set()

    for obs_date, flow in record:
        if obs_date.year == target_year:
            continue
        obs_doy = obs_date.timetuple().tm_yday
        # Handle year-end wrap: compare circularly within 365-day calendar
        diff = abs(obs_doy - target_doy)
        diff = min(diff, 365 - diff)
        if diff <= window_days:
            values.append(flow)
            years_seen.add(obs_date.year)

    if len(years_seen) < min_years:
        return None

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _percentile(p: float) -> float:
        idx = p / 100.0 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])

    all_dates = [d for d, _ in record]
    return FlowClimatology(
        site_id=site_id,
        day_of_year=target_doy,
        window_days=window_days,
        years_of_data=len(years_seen),
        n_values=n,
        p05=round(_percentile(5), 3),
        p10=round(_percentile(10), 3),
        p25=round(_percentile(25), 3),
        p50=round(_percentile(50), 3),
        p75=round(_percentile(75), 3),
        p90=round(_percentile(90), 3),
        p95=round(_percentile(95), 3),
        record_start=min(all_dates),
        record_end=max(all_dates),
    )


def flow_percentile(
    value: float,
    record: list[tuple[date, float]],
    target: date,
    window_days: int = 7,
) -> float | None:
    """Return empirical percentile (0–100) of value within day-of-year window.

    Uses all years except the target year. Returns None if no matching values.
    """
    target_doy = target.timetuple().tm_yday
    target_year = target.year

    values: list[float] = []
    for obs_date, flow in record:
        if obs_date.year == target_year:
            continue
        obs_doy = obs_date.timetuple().tm_yday
        diff = abs(obs_doy - target_doy)
        diff = min(diff, 365 - diff)
        if diff <= window_days:
            values.append(flow)

    if not values:
        return None

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    below = sum(1 for v in sorted_vals if v < value)
    equal = sum(1 for v in sorted_vals if v == value)
    return round((below + 0.5 * equal) / n * 100.0, 1)


def compute_deviation(current: float, baseline_mean: float, baseline_std: float) -> float:
    """Compute z-score deviation from baseline. Capped at +/-3."""
    if baseline_std is None or baseline_std < 1e-6:
        diff = current - baseline_mean
        return max(-3.0, min(3.0, diff / max(abs(baseline_mean), 1e-6) * 3.0))
    return max(-3.0, min(3.0, (current - baseline_mean) / baseline_std))


def compute_health_score(deviations: list[float]) -> float:
    """Compute ecosystem health score (0-100) from deviations.

    100 = all signals at baseline. RMS-scaled so 2-sigma = score 50.
    """
    if not deviations:
        return 50.0
    rms = math.sqrt(sum(d * d for d in deviations) / len(deviations))
    score = max(0.0, min(100.0, 100.0 * (1.0 - rms / 4.0)))
    return round(score, 1)


def classify_trend(health_scores: list[float]) -> str:
    """Classify trend from recent health scores (most recent last)."""
    if len(health_scores) < 3:
        return "stable"
    recent = health_scores[-3:]
    slope = (recent[-1] - recent[0]) / 2.0

    if recent[-1] < 25:
        return "critical"
    elif slope > 3.0:
        return "improving"
    elif slope < -3.0:
        return "declining"
    return "stable"


def _ols_slope(xs: list[float], ys: list[float]) -> float:
    """OLS slope of ys ~ xs."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 1e-12 else 0.0


async def compute_baselines_from_era5(
    era5_adapter,
    lat: float,
    lng: float,
    target_date: datetime,
    years_back: int = 30,
) -> BaselineValues:
    """Compute climate baselines from ERA5 historical data.

    Uses a ±7-day window around the target day-of-year across years_back prior
    years. Defaults to 30 years for a proper climate normal. Also computes a
    linear temperature trend (°C/decade) via OLS over the yearly window means.
    """
    doy = target_date.timetuple().tm_yday
    location_id = f"location:{lat:.2f}_{lng:.2f}"
    temps: list[float] = []
    precips: list[float] = []
    yearly_temp_means: list[tuple[int, float]] = []  # (year, mean_temp)

    for year_offset in range(1, years_back + 1):
        year = target_date.year - year_offset
        center = datetime(year, 1, 1) + timedelta(days=doy - 1)
        start = (center - timedelta(days=7)).strftime("%Y-%m-%d")
        end = (center + timedelta(days=7)).strftime("%Y-%m-%d")

        try:
            era5_data = await era5_adapter.get_daily(lat=lat, lng=lng, start_date=start, end_date=end)
            daily = era5_data.get("daily", {})
            year_temps = [v for v in daily.get("temperature_2m_mean", []) if v is not None]
            if year_temps:
                temps.extend(year_temps)
                yearly_temp_means.append((year, sum(year_temps) / len(year_temps)))
            if "precipitation_sum" in daily:
                precips.extend([v for v in daily["precipitation_sum"] if v is not None])
        except Exception:
            continue

    earliest_year = target_date.year - years_back
    latest_year = target_date.year - 1
    baseline = BaselineValues(
        location_id=location_id,
        day_of_year=doy,
        years_of_data=len(yearly_temp_means) or years_back,
        baseline_period=f"{earliest_year}–{latest_year}",
    )

    if temps:
        baseline.temp_mean_c = round(sum(temps) / len(temps), 2)
        if len(temps) > 1:
            mean = baseline.temp_mean_c
            baseline.temp_std_c = round(math.sqrt(sum((t - mean) ** 2 for t in temps) / (len(temps) - 1)), 2)
    if precips:
        baseline.precip_mean_mm = round(sum(precips) / len(precips), 2)
        if len(precips) > 1:
            mean = baseline.precip_mean_mm
            baseline.precip_std_mm = round(math.sqrt(sum((p - mean) ** 2 for p in precips) / (len(precips) - 1)), 2)

    if len(yearly_temp_means) >= 3:
        years_xs = [float(y) for y, _ in yearly_temp_means]
        means_ys = [m for _, m in yearly_temp_means]
        slope_per_year = _ols_slope(years_xs, means_ys)
        baseline.temp_trend_c_per_decade = round(slope_per_year * 10.0, 3)

    return baseline


async def compute_baselines_from_usgs(
    usgs_adapter,
    lat: float,
    lng: float,
    target_date: datetime,
    radius_km: float = 50.0,
    min_years: int = 10,
    current_flow: float | None = None,
) -> dict:
    """Compute day-of-year streamflow baseline from a single nearby USGS gauge.

    Picks the nearest gauge with a daily discharge record of at least min_years,
    builds day-of-year climatology, and returns percentile-aware baseline keys.

    Returns {"baseline_kind": "insufficient_record"} when no gauge qualifies.
    Compatible keys (streamflow_mean_cfs, streamflow_std_cfs) are retained so
    existing callers continue to work.
    """
    from .schema import SearchParams

    # Find nearby gauges (sorted by distance, nearest first)
    try:
        nearby_sites = await usgs_adapter._find_sites_near_public(lat, lng, radius_km)
    except AttributeError:
        # Adapter doesn't expose the helper; fall back to search-based discovery
        nearby_sites = []
        start = (target_date - timedelta(days=30)).strftime("%Y-%m-%d")
        end = target_date.strftime("%Y-%m-%d")
        try:
            obs = await usgs_adapter.search(
                SearchParams(lat=lat, lng=lng, radius_km=radius_km, start_date=start, end_date=end, limit=20)
            )
            seen: set[str] = set()
            for o in obs:
                sid = o.location.site_id or ""
                if sid and sid not in seen:
                    seen.add(sid)
                    nearby_sites.append({"site_id": f"USGS-{sid}"})
        except Exception:
            pass

    if not nearby_sites:
        return {"baseline_kind": "insufficient_record"}

    # Try gauges nearest-first; pick first with enough years of daily record
    for site_info in nearby_sites[:5]:
        site_id = site_info.get("site_id") or site_info.get("id", "")
        if not site_id:
            continue
        try:
            record = await usgs_adapter.get_daily_record(site_id, parameter_code="00060")
        except Exception:
            continue

        if not record:
            continue

        target_as_date = target_date.date() if isinstance(target_date, datetime) else target_date
        clim = flow_climatology(record, target_as_date, min_years=min_years, site_id=site_id)
        if clim is None:
            continue

        result: dict = {
            "baseline_kind": "doy_record",
            "site_id": site_id,
            "streamflow_mean_cfs": clim.p50,
            "streamflow_std_cfs": round((clim.p75 - clim.p25) / 1.349, 2),
            "climatology": clim.model_dump(),
        }
        if current_flow is not None:
            pct = flow_percentile(current_flow, record, target_as_date)
            if pct is not None:
                result["streamflow_percentile"] = pct
        return result

    return {"baseline_kind": "insufficient_record"}
