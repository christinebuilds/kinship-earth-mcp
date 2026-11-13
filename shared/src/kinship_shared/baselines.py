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

    streamflow_mean_cfs: Optional[float] = None
    streamflow_std_cfs: Optional[float] = None

    species_richness_mean: Optional[int] = None
    ndvi_mean: Optional[float] = None
    ndvi_std: Optional[float] = None


class FlowClimatology(BaseModel):
    """A gauge's own record for one time of year: what 'normal' flow means here."""

    site_id: str
    day_of_year: int
    window_days: int = Field(description="Half-width of the day-of-year window")
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


def _doy_distance(a: date, target: date) -> int:
    """Days between two dates' positions in the year, wrapping at year end."""
    d = abs(a.timetuple().tm_yday - target.timetuple().tm_yday)
    return min(d, 366 - d)


def _window_values(
    record: list[tuple[date, float]], target: date, window_days: int
) -> list[tuple[date, float]]:
    """Values within ±window_days of the target day-of-year, from every year
    except the target's own."""
    return [
        (d, v) for d, v in record
        if d.year != target.year and v is not None and _doy_distance(d, target) <= window_days
    ]


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile of an already sorted list, q in [0, 1]."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def flow_climatology(
    record: list[tuple[date, float]],
    target: date,
    window_days: int = 7,
    min_years: int = 10,
    site_id: str = "",
) -> Optional[FlowClimatology]:
    """Day-of-year flow percentiles from a gauge's daily record.

    Returns None when fewer than ``min_years`` distinct years contribute.
    There is deliberately no fallback to a shorter window.
    """
    window = _window_values(record, target, window_days)
    years = {d.year for d, _ in window}
    if len(years) < min_years:
        return None
    values = sorted(v for _, v in window)
    q = lambda x: round(_quantile(values, x), 3)  # noqa: E731
    return FlowClimatology(
        site_id=site_id,
        day_of_year=target.timetuple().tm_yday,
        window_days=window_days,
        years_of_data=len(years),
        n_values=len(values),
        p05=q(0.05), p10=q(0.10), p25=q(0.25), p50=q(0.50),
        p75=q(0.75), p90=q(0.90), p95=q(0.95),
        record_start=min(d for d, _ in record),
        record_end=max(d for d, _ in record),
    )


def flow_percentile(
    value: float,
    record: list[tuple[date, float]],
    target: date,
    window_days: int = 7,
) -> Optional[float]:
    """Empirical percentile (0–100) of ``value`` among the record's values for
    this time of year. 0 = below everything on record, 100 = above everything.
    Returns None if the record has no values in the window."""
    values = [v for _, v in _window_values(record, target, window_days)]
    if not values:
        return None
    below = sum(1 for v in values if v < value)
    equal = sum(1 for v in values if v == value)
    return round(100.0 * (below + 0.5 * equal) / len(values), 1)


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


async def compute_baselines_from_era5(
    era5_adapter,
    lat: float,
    lng: float,
    target_date: datetime,
    years_back: int = 5,
) -> BaselineValues:
    """Compute climate baselines from ERA5 historical data."""
    from .schema import SearchParams

    doy = target_date.timetuple().tm_yday
    location_id = f"location:{lat:.2f}_{lng:.2f}"
    temps: list[float] = []
    precips: list[float] = []

    for year_offset in range(1, years_back + 1):
        year = target_date.year - year_offset
        center = datetime(year, 1, 1) + timedelta(days=doy - 1)
        start = (center - timedelta(days=7)).strftime("%Y-%m-%d")
        end = (center + timedelta(days=7)).strftime("%Y-%m-%d")

        try:
            era5_data = await era5_adapter.get_daily(lat=lat, lng=lng, start_date=start, end_date=end)
            daily = era5_data.get("daily", {})
            if "temperature_2m_mean" in daily:
                temps.extend([v for v in daily["temperature_2m_mean"] if v is not None])
            if "precipitation_sum" in daily:
                precips.extend([v for v in daily["precipitation_sum"] if v is not None])
        except Exception:
            continue

    baseline = BaselineValues(location_id=location_id, day_of_year=doy, years_of_data=years_back)

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

    return baseline


async def compute_baselines_from_usgs(
    usgs_adapter,
    lat: float,
    lng: float,
    target_date: datetime,
    radius_km: float = 50.0,
) -> dict:
    """Compute streamflow baselines from nearby USGS gauges."""
    from .schema import SearchParams

    flows: list[float] = []
    start = (target_date - timedelta(days=30)).strftime("%Y-%m-%d")
    end = target_date.strftime("%Y-%m-%d")

    try:
        observations = await usgs_adapter.search(
            SearchParams(lat=lat, lng=lng, radius_km=radius_km, start_date=start, end_date=end, limit=100)
        )
        for obs in observations:
            if obs.value and isinstance(obs.value, dict):
                flow = obs.value.get("discharge_cfs") or obs.value.get("value")
                if flow is not None:
                    try:
                        flows.append(float(flow))
                    except (ValueError, TypeError):
                        continue
    except Exception:
        pass

    result: dict = {}
    if flows:
        result["streamflow_mean_cfs"] = round(sum(flows) / len(flows), 2)
        if len(flows) > 1:
            mean = result["streamflow_mean_cfs"]
            result["streamflow_std_cfs"] = round(
                math.sqrt(sum((f - mean) ** 2 for f in flows) / (len(flows) - 1)), 2
            )
    return result
