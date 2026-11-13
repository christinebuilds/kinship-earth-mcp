"""
Baseline computation for ecosystem monitoring.

Computes historical normals from ERA5 (climate) and USGS NWIS (hydrology)
to establish what 'normal' looks like for a location at a given time of year.
Baselines are the reference against which anomalies are detected.
"""

from __future__ import annotations

import math
from datetime import date as date_type
from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel, Field


class BaselineValues(BaseModel):
    """Historical normal values for a location at a specific day-of-year."""

    location_id: str = Field(description="e.g. 'location:41.50_-70.70'")
    day_of_year: int = Field(ge=1, le=366)
    years_of_data: int = Field(default=0)

    temp_mean_c: Optional[float] = None
    temp_std_c: Optional[float] = None
    temp_trend_c_per_decade: Optional[float] = None
    baseline_period: Optional[str] = None
    precip_mean_mm: Optional[float] = None
    precip_std_mm: Optional[float] = None

    streamflow_mean_cfs: Optional[float] = None
    streamflow_std_cfs: Optional[float] = None

    species_richness_mean: Optional[int] = None
    ndvi_mean: Optional[float] = None
    ndvi_std: Optional[float] = None


class FlowClimatology(BaseModel):
    """Day-of-year flow statistics from a long daily record for one gauge."""

    site_id: str
    day_of_year: int
    window_days: int = 7
    years_of_data: int
    n_values: int
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    record_start: date_type
    record_end: date_type


def _doy_distance(doy1: int, doy2: int) -> int:
    """Minimum circular distance between two day-of-year values."""
    diff = abs(doy1 - doy2)
    return min(diff, 366 - diff)


def _percentile_sorted(sorted_vals: list[float], p: float) -> float:
    """Linear interpolation percentile from a sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def _window_values(
    record: list[tuple[date_type, float]],
    target: date_type,
    window_days: int,
) -> list[float]:
    """Values within ±window_days of target DOY, excluding the target year."""
    target_doy = target.timetuple().tm_yday
    return [
        v for d, v in record
        if d.year != target.year and _doy_distance(d.timetuple().tm_yday, target_doy) <= window_days
    ]


def flow_climatology(
    record: list[tuple[date_type, float]],
    target: date_type,
    window_days: int = 7,
    min_years: int = 10,
) -> FlowClimatology | None:
    """
    Day-of-year flow climatology from a full daily record.

    Returns None if fewer than min_years distinct years contribute values
    within the DOY window. Never silently falls back to a shorter window.
    """
    if not record:
        return None

    target_doy = target.timetuple().tm_yday
    window_records = [
        (d, v) for d, v in record
        if d.year != target.year and _doy_distance(d.timetuple().tm_yday, target_doy) <= window_days
    ]
    contributing_years = {d.year for d, _ in window_records}
    if len(contributing_years) < min_years:
        return None

    values = [v for _, v in window_records]
    sorted_vals = sorted(values)
    all_dates = [d for d, _ in record]

    return FlowClimatology(
        site_id="",  # filled by compute_baselines_from_usgs
        day_of_year=target_doy,
        window_days=window_days,
        years_of_data=len(contributing_years),
        n_values=len(values),
        p05=round(_percentile_sorted(sorted_vals, 5), 2),
        p10=round(_percentile_sorted(sorted_vals, 10), 2),
        p25=round(_percentile_sorted(sorted_vals, 25), 2),
        p50=round(_percentile_sorted(sorted_vals, 50), 2),
        p75=round(_percentile_sorted(sorted_vals, 75), 2),
        p90=round(_percentile_sorted(sorted_vals, 90), 2),
        p95=round(_percentile_sorted(sorted_vals, 95), 2),
        record_start=min(all_dates),
        record_end=max(all_dates),
    )


def flow_percentile(
    value: float,
    record: list[tuple[date_type, float]],
    target: date_type,
    window_days: int = 7,
) -> float | None:
    """
    Empirical percentile (0–100) of value in the DOY window of the record.

    Returns 0.0 when value is at or below all window values, 100.0 when at
    or above all window values. Returns None if the window is empty.
    """
    values = _window_values(record, target, window_days)
    if not values:
        return None
    count_less = sum(1 for v in values if v < value)
    return round(max(0.0, min(100.0, count_less / len(values) * 100.0)), 1)


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


def _ols_slope(xs: list[float], ys: list[float]) -> float | None:
    """Ordinary least squares slope, or None if fewer than 2 points."""
    n = len(xs)
    if n < 2:
        return None
    sx = sum(xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    return (n * sxy - sx * sy) / denom


async def compute_baselines_from_era5(
    era5_adapter,
    lat: float,
    lng: float,
    target_date: datetime,
    years_back: int = 30,
) -> BaselineValues:
    """
    Compute climate baselines from ERA5 historical data.

    Fetches one ±7-day window per year for years_back prior years.
    Open-Meteo (ERA5 archive) supports arbitrary date ranges; per-year
    calls are used here to keep payloads small and DOY windowing explicit.
    Default changed from 5 to 30 years for spec 015 (honest baselines).
    """
    doy = target_date.timetuple().tm_yday
    location_id = f"location:{lat:.2f}_{lng:.2f}"
    temps: list[float] = []
    precips: list[float] = []
    yearly_temp_means: list[tuple[int, float]] = []  # (year, mean)

    for year_offset in range(1, years_back + 1):
        year = target_date.year - year_offset
        center = datetime(year, 1, 1) + timedelta(days=doy - 1)
        start = (center - timedelta(days=7)).strftime("%Y-%m-%d")
        end = (center + timedelta(days=7)).strftime("%Y-%m-%d")

        try:
            era5_data = await era5_adapter.get_daily(lat=lat, lng=lng, start_date=start, end_date=end)
            daily = era5_data.get("daily", {})
            year_temps = [v for v in daily.get("temperature_2m_mean", []) if v is not None]
            year_precips = [v for v in daily.get("precipitation_sum", []) if v is not None]
            temps.extend(year_temps)
            precips.extend(year_precips)
            if year_temps:
                yearly_temp_means.append((year, sum(year_temps) / len(year_temps)))
        except Exception:
            continue

    earliest_year = target_date.year - years_back
    baseline = BaselineValues(
        location_id=location_id,
        day_of_year=doy,
        years_of_data=len(yearly_temp_means),
        baseline_period=f"{earliest_year}–{target_date.year - 1}" if yearly_temp_means else None,
    )

    if temps:
        baseline.temp_mean_c = round(sum(temps) / len(temps), 2)
        if len(temps) > 1:
            mean = baseline.temp_mean_c
            baseline.temp_std_c = round(math.sqrt(sum((t - mean) ** 2 for t in temps) / (len(temps) - 1)), 2)
    if len(yearly_temp_means) >= 2:
        xs = [float(yr) for yr, _ in yearly_temp_means]
        ys = [m for _, m in yearly_temp_means]
        slope = _ols_slope(xs, ys)
        if slope is not None:
            baseline.temp_trend_c_per_decade = round(slope * 10, 3)

    if precips:
        baseline.precip_mean_mm = round(sum(precips) / len(precips), 2)
        if len(precips) > 1:
            mean = baseline.precip_mean_mm
            baseline.precip_std_mm = round(math.sqrt(sum((p - mean) ** 2 for p in precips) / (len(precips) - 1)), 2)

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
    """
    Compute day-of-year streamflow baseline from a single nearby USGS gauge.

    Picks the nearest gauge with ≥ min_years of daily discharge record.
    Returns the day-of-year median as streamflow_mean_cfs (for compatibility)
    and the IQR-based spread as streamflow_std_cfs, plus site_id,
    streamflow_percentile (of current_flow when provided), climatology dict,
    and baseline_kind: "doy_record".

    Returns {"baseline_kind": "insufficient_record"} when no gauge qualifies.
    Existing callers should check baseline_kind before using flow values.
    """
    from .schema import SearchParams

    if not hasattr(usgs_adapter, "get_daily_record"):
        return {}

    # Find sites near the target location
    try:
        sites = await usgs_adapter._find_sites_near_public(lat, lng, radius_km)
    except Exception:
        return {"baseline_kind": "insufficient_record"}

    if not sites:
        return {"baseline_kind": "insufficient_record"}

    target = target_date.date() if hasattr(target_date, "date") else target_date

    for site_feature in sites:
        site_props = site_feature.get("properties", {})
        site_id = site_props.get("id", "")
        if not site_id:
            continue

        try:
            record = await usgs_adapter.get_daily_record(site_id)
        except Exception:
            continue

        if not record:
            continue

        clim = flow_climatology(record, target, min_years=min_years)
        if clim is None:
            continue

        clim.site_id = site_id

        # IQR-based spread (comparable to std but robust for skewed distributions)
        spread = (clim.p75 - clim.p25) / 1.349

        result: dict = {
            "baseline_kind": "doy_record",
            "site_id": site_id,
            "streamflow_mean_cfs": clim.p50,
            "streamflow_std_cfs": round(spread, 2),
            "climatology": clim.model_dump(),
        }

        if current_flow is not None:
            pct = flow_percentile(current_flow, record, target)
            if pct is not None:
                result["streamflow_percentile"] = pct

        return result

    return {"baseline_kind": "insufficient_record"}
