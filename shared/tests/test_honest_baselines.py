"""Spec 015: single-gauge flow baseline and 30-year climate normals (offline, faked adapters)."""

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from kinship_shared.baselines import compute_baselines_from_era5, compute_baselines_from_usgs

TARGET = datetime(2026, 11, 14, tzinfo=timezone.utc)


def _seasonal_record(start_year: int, end: date, scale: float = 1.0) -> list[tuple[date, float]]:
    d, out = date(start_year, 1, 1), []
    while d <= end:
        doy = d.timetuple().tm_yday
        out.append((d, scale * (500 + 400 * math.cos(2 * math.pi * (doy - 15) / 365)) * (1 + (d.year % 7) / 20)))
        d += timedelta(days=1)
    return out


class FakeUSGS:
    def __init__(self, gauges, records):
        self.gauges, self.records, self.fetched = gauges, records, []

    async def find_discharge_gauges(self, lat, lng, radius_km=50.0):
        return self.gauges

    async def get_daily_record(self, site_id, parameter_code="00060", start_date=None, end_date=None):
        self.fetched.append((site_id, start_date))
        rec = self.records[site_id]
        if start_date:
            s, e = date.fromisoformat(start_date), date.fromisoformat(end_date)
            rec = [(d, v) for d, v in rec if s <= d <= e]
        return rec


def _gauge(site_id, km, begin, end=date(2026, 11, 12)):
    return {"site_id": site_id, "distance_km": km, "begin": begin, "end": end, "lat": 0, "lng": 0}


async def test_skips_short_record_and_uses_one_gauge():
    near_short = _gauge("USGS-SHORT", 1.0, date(2020, 1, 1))
    far_long = _gauge("USGS-LONG", 8.0, date(1950, 1, 1))
    other_long = _gauge("USGS-OTHER", 9.0, date(1950, 1, 1))
    usgs = FakeUSGS([near_short, far_long, other_long], {
        "USGS-LONG": _seasonal_record(1950, date(2026, 11, 12)),
        "USGS-OTHER": _seasonal_record(1950, date(2026, 11, 12), scale=100),
    })
    res = await compute_baselines_from_usgs(usgs, 38.5, -122.9, TARGET)
    assert res["baseline_kind"] == "doy_record"
    assert res["site_id"] == "USGS-LONG"
    assert {s for s, _ in usgs.fetched} == {"USGS-LONG"}  # never pooled with OTHER
    assert res["climatology"]["years_of_data"] >= 70
    assert res["streamflow_mean_cfs"] == res["climatology"]["p50"]
    assert res["current_flow_date"] == "2026-11-12"
    assert 0 <= res["streamflow_percentile"] <= 100


async def test_current_flow_given_sets_percentile():
    usgs = FakeUSGS([_gauge("USGS-LONG", 2.0, date(1950, 1, 1))],
                    {"USGS-LONG": _seasonal_record(1950, date(2026, 11, 12))})
    res = await compute_baselines_from_usgs(usgs, 38.5, -122.9, TARGET, current_flow=1.0)
    assert res["current_flow_cfs"] == 1.0
    assert res["streamflow_percentile"] == 0.0


async def test_insufficient_record_has_no_streamflow_keys():
    usgs = FakeUSGS([_gauge("USGS-SHORT", 1.0, date(2020, 1, 1))], {})
    res = await compute_baselines_from_usgs(usgs, 38.5, -122.9, TARGET)
    assert res == {"baseline_kind": "insufficient_record"}


async def test_inactive_gauge_skipped():
    usgs = FakeUSGS([_gauge("USGS-OLD", 1.0, date(1950, 1, 1), end=date(1990, 1, 1))], {})
    res = await compute_baselines_from_usgs(usgs, 38.5, -122.9, TARGET)
    assert res["baseline_kind"] == "insufficient_record"


class FakeERA5:
    def __init__(self, warming_per_year=0.03):
        self.calls, self.w = [], warming_per_year

    async def get_daily(self, lat, lng, start_date, end_date, variables=None):
        self.calls.append((start_date, end_date))
        s, e = date.fromisoformat(start_date), date.fromisoformat(end_date)
        times, temps, precs = [], [], []
        d = s
        while d <= e:
            times.append(d.isoformat())
            temps.append(12.0 + self.w * (d.year - 1996))
            precs.append(2.0)
            d += timedelta(days=1)
        return {"daily": {"time": times, "temperature_2m_mean": temps, "precipitation_sum": precs}}


async def test_era5_thirty_years_one_call_with_trend():
    era5 = FakeERA5(warming_per_year=0.03)
    b = await compute_baselines_from_era5(era5, 38.5, -122.9, TARGET)
    assert len(era5.calls) == 1
    assert b.years_of_data == 30
    assert b.baseline_period == "1996–2025"
    assert b.temp_trend_c_per_decade == pytest.approx(0.3, abs=0.01)
    assert b.precip_mean_mm == 2.0


async def test_era5_failure_returns_empty_baseline():
    class Broken:
        async def get_daily(self, **kw):
            raise RuntimeError("down")
    b = await compute_baselines_from_era5(Broken(), 38.5, -122.9, TARGET)
    assert b.temp_mean_c is None and b.years_of_data == 0


@pytest.mark.network
async def test_usgs_daily_record_reachable_long_record():
    """Live: Russian River near Guerneville has decades of daily means, oldest first."""
    from usgs_nwis_mcp.adapter import USGSNWISAdapter
    rec = await USGSNWISAdapter().get_daily_record("USGS-11467000")
    assert len({d.year for d, _ in rec}) > 10
    assert [d for d, _ in rec] == sorted(d for d, _ in rec)
