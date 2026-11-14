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

    temp_trend_c_per_decade: Optional[float] = Field(
        default=None, description="OLS slope of the yearly window-mean temperatures"
    )
    baseline_period: Optional[str] = Field(default=None, description="e.g. '1996–2025'")

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


def _ols_slope(xs: list[float], ys: list[float]) -> Optional[float]:
    """Least-squares slope of ys on xs; None with fewer than 3 points."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def _std(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


async def compute_baselines_from_era5(
    era5_adapter,
    lat: float,
    lng: float,
    target_date: datetime,
    years_back: int = 30,
    window_days: int = 7,
) -> BaselineValues:
    """Climate normals: ±window_days around this day-of-year over the
    ``years_back`` years before the target year.

    The Open-Meteo archive accepts a multi-decade range in one request
    (31 years answered in under 2 s when checked 2026-11-14), so we fetch
    once and slice by day-of-year rather than making one call per year.
    """
    doy = target_date.timetuple().tm_yday
    location_id = f"location:{lat:.2f}_{lng:.2f}"
    first_year = target_date.year - years_back
    last_year = target_date.year - 1

    def center(year: int) -> date:
        return date(year, 1, 1) + timedelta(days=min(doy, 365) - 1)

    start = center(first_year) - timedelta(days=window_days)
    end = center(last_year) + timedelta(days=window_days)

    baseline = BaselineValues(location_id=location_id, day_of_year=doy, years_of_data=0)
    try:
        era5_data = await era5_adapter.get_daily(
            lat=lat, lng=lng, start_date=start.isoformat(), end_date=end.isoformat()
        )
    except Exception:
        return baseline

    daily = era5_data.get("daily", {}) if isinstance(era5_data, dict) else {}
    times = daily.get("time") or []
    temp_by_day = dict(zip(times, daily.get("temperature_2m_mean") or []))
    precip_by_day = dict(zip(times, daily.get("precipitation_sum") or []))

    temps: list[float] = []
    precips: list[float] = []
    yearly_temp: list[tuple[int, float]] = []
    years_with_data: set[int] = set()
    for year in range(first_year, last_year + 1):
        c = center(year)
        year_temps = []
        for k in range(-window_days, window_days + 1):
            key = (c + timedelta(days=k)).isoformat()
            t = temp_by_day.get(key)
            p = precip_by_day.get(key)
            if t is not None:
                year_temps.append(t)
            if p is not None:
                precips.append(p)
                years_with_data.add(year)
        if year_temps:
            temps.extend(year_temps)
            yearly_temp.append((year, sum(year_temps) / len(year_temps)))
            years_with_data.add(year)

    baseline.years_of_data = len(years_with_data)
    if years_with_data:
        baseline.baseline_period = f"{min(years_with_data)}–{max(years_with_data)}"
    if temps:
        baseline.temp_mean_c = round(sum(temps) / len(temps), 2)
        sd = _std(temps)
        baseline.temp_std_c = round(sd, 2) if sd is not None else None
        slope = _ols_slope([y for y, _ in yearly_temp], [m for _, m in yearly_temp])
        if slope is not None:
            baseline.temp_trend_c_per_decade = round(slope * 10, 3)
    if precips:
        baseline.precip_mean_mm = round(sum(precips) / len(precips), 2)
        sd = _std(precips)
        baseline.precip_std_mm = round(sd, 2) if sd is not None else None
    return baseline


async def compute_baselines_from_usgs(
    usgs_adapter,
    lat: float,
    lng: float,
    target_date: datetime,
    radius_km: float = 50.0,
    current_flow: Optional[float] = None,
    min_years: int = 10,
    window_days: int = 7,
    max_candidates: int = 5,
) -> dict:
    """Streamflow baseline from ONE gauge's own day-of-year record.

    Picks the nearest gauge whose daily discharge record spans at least
    ``min_years`` and that still reports, and compares against that gauge
    only. Gauges are never pooled. Returns ``{"baseline_kind":
    "insufficient_record"}`` (and no streamflow keys) when no nearby gauge
    has enough record; callers then skip flow anomaly detection.

    Keys kept for older callers: ``streamflow_mean_cfs`` is now the
    day-of-year median (p50) and ``streamflow_std_cfs`` the IQR / 1.349.
    When ``current_flow`` is not given, the gauge's latest daily mean within
    7 days of the target is used and returned as ``current_flow_cfs``.
    """
    insufficient = {"baseline_kind": "insufficient_record"}
    target = target_date.date() if isinstance(target_date, datetime) else target_date

    try:
        gauges = await usgs_adapter.find_discharge_gauges(lat, lng, radius_km)
    except Exception:
        return insufficient

    for gauge in gauges:
        if max_candidates <= 0:
            break
        span_years = (gauge["end"] - gauge["begin"]).days / 365.25
        if span_years < min_years or (target - gauge["end"]).days > 30:
            continue
        max_candidates -= 1
        try:
            record = await usgs_adapter.get_daily_record(gauge["site_id"])
        except Exception:
            continue
        clim = flow_climatology(record, target, window_days=window_days,
                                min_years=min_years, site_id=gauge["site_id"])
        if clim is None:
            continue

        result: dict = {
            "baseline_kind": "doy_record",
            "site_id": gauge["site_id"],
            "gauge_distance_km": gauge.get("distance_km"),
            "streamflow_mean_cfs": clim.p50,
            "streamflow_std_cfs": round((clim.p75 - clim.p25) / 1.349, 3),
            "climatology": clim.model_dump(mode="json"),
        }

        flow, flow_date = current_flow, None
        if flow is None:
            # The cached long record may be days old; fetch the recent tail.
            recent_start = (target - timedelta(days=7)).isoformat()
            try:
                recent = await usgs_adapter.get_daily_record(
                    gauge["site_id"], start_date=recent_start, end_date=target.isoformat()
                )
            except Exception:
                recent = []
            recent = recent or [(d, v) for d, v in record if 0 <= (target - d).days <= 7]
            if recent:
                flow_date, flow = recent[-1]
        if flow is not None:
            result["current_flow_cfs"] = flow
            if flow_date is not None:
                result["current_flow_date"] = flow_date.isoformat()
            result["streamflow_percentile"] = flow_percentile(
                flow, record, target, window_days=window_days
            )
        return result

    return insufficient
