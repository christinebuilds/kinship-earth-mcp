"""Tests for day-of-year flow climatology (spec 015). Offline, synthetic records."""

import math
from datetime import date, timedelta

from kinship_shared.baselines import flow_climatology, flow_percentile


def _seasonal(d: date) -> float:
    """Wet winters, dry summers: peak ~Feb 1, trough ~Aug 1."""
    return 500.0 + 400.0 * math.cos(2 * math.pi * (d.timetuple().tm_yday - 32) / 365.25)


def _record(start_year: int, end_year: int, fn=_seasonal) -> list[tuple[date, float]]:
    out, d = [], date(start_year, 1, 1)
    while d.year <= end_year:
        out.append((d, fn(d)))
        d += timedelta(days=1)
    return out


def test_median_matches_seasonal_cycle():
    record = _record(1980, 2019)
    target = date(2020, 8, 1)
    clim = flow_climatology(record, target, site_id="USGS-TEST")
    assert clim is not None
    assert clim.years_of_data == 40
    assert clim.n_values == 40 * 15
    assert abs(clim.p50 - _seasonal(target)) / _seasonal(target) < 0.05
    assert clim.p05 <= clim.p25 <= clim.p50 <= clim.p75 <= clim.p95
    assert clim.record_start == date(1980, 1, 1) and clim.record_end == date(2019, 12, 31)


def test_year_end_wrap_includes_late_december():
    # Only December has data; a Jan 3 target must still find Dec 27–31.
    record = [(d, v) for d, v in _record(1990, 2010) if d.month == 12]
    clim = flow_climatology(record, date(2011, 1, 3))
    assert clim is not None
    # 4 or 5 late-December days per year (leap years shift day-of-year by one)
    assert 21 * 4 <= clim.n_values <= 21 * 5
    assert clim.years_of_data == 21


def test_target_year_excluded():
    record = _record(1990, 2010, fn=lambda d: 100.0)
    # A flood year that must not leak into its own baseline.
    record = [(d, 10_000.0 if d.year == 2005 else v) for d, v in record]
    clim = flow_climatology(record, date(2005, 6, 15))
    assert clim.years_of_data == 20
    assert clim.p95 == 100.0
    assert flow_percentile(10_000.0, record, date(2005, 6, 15)) == 100.0


def test_too_few_years_returns_none():
    record = _record(2012, 2020)  # 9 years
    assert flow_climatology(record, date(2021, 6, 15)) is None
    assert flow_climatology(record, date(2021, 6, 15), min_years=9) is not None


def test_flow_percentile():
    # 20 years, one value per year on the target day: 1..20 * 10 cfs
    target = date(2021, 6, 15)
    record = [(date(2000 + i, 6, 15), (i + 1) * 10.0) for i in range(20)]
    assert flow_percentile(0.0, record, target) == 0.0
    assert flow_percentile(1e6, record, target) == 100.0
    assert abs(flow_percentile(15.0, record, target) - 5.0) < 0.01  # above 1 of 20
    assert abs(flow_percentile(105.0, record, target) - 50.0) < 0.01
    assert flow_percentile(50.0, [], target) is None
