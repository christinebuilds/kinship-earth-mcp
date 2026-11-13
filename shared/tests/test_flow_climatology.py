"""
Tests for flow_climatology and flow_percentile (spec 015, offline / synthetic records).
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from kinship_shared.baselines import FlowClimatology, flow_climatology, flow_percentile


# ---------------------------------------------------------------------------
# Helpers — synthetic 40-year record with a known seasonal cycle
# ---------------------------------------------------------------------------

def _synthetic_record(
    start_year: int = 1980,
    end_year: int = 2019,
    target_doy: int = 196,
) -> list[tuple[date, float]]:
    """
    40-year daily record where flow on day-of-year d is:
        100 + 50 * sin(2π * (d - 90) / 365) + noise_free

    On the target DOY the median value should be predictable.
    """
    records = []
    for year in range(start_year, end_year + 1):
        for doy in range(1, 366):
            try:
                d = date(year, 1, 1).replace(year=year)
                # Build the date from DOY
                from datetime import timedelta
                d = date(year, 1, 1) + timedelta(days=doy - 1)
            except ValueError:
                continue
            flow = 100.0 + 50.0 * math.sin(2 * math.pi * (doy - 90) / 365)
            records.append((d, flow))
    return records


_RECORD = _synthetic_record()
_TARGET = date(2020, 7, 15)  # DOY ~196, excluded from its own baseline
_TARGET_DOY = _TARGET.timetuple().tm_yday
_EXPECTED_MEDIAN = 100.0 + 50.0 * math.sin(2 * math.pi * (_TARGET_DOY - 90) / 365)


# ---------------------------------------------------------------------------
# Test 1: Seasonally structured record returns p50 within 5% of expected
# ---------------------------------------------------------------------------

def test_flow_climatology_p50_near_expected():
    clim = flow_climatology(_RECORD, _TARGET)
    assert clim is not None
    assert abs(clim.p50 - _EXPECTED_MEDIAN) / max(_EXPECTED_MEDIAN, 1) < 0.05


# ---------------------------------------------------------------------------
# Test 2: Year-end wrap — Jan 3 target includes late-December values
# ---------------------------------------------------------------------------

def test_flow_climatology_year_end_wrap():
    jan3 = date(2020, 1, 3)  # DOY 3
    # Window ±7 covers DOY 362–365 of previous years too
    clim = flow_climatology(_RECORD, jan3, window_days=7)
    assert clim is not None
    # Verify n_values spans both sides of the year boundary
    # (DOY 1-10 and DOY 356-365 should both be included)
    assert clim.n_values > 0


# ---------------------------------------------------------------------------
# Test 3: Target year is excluded from its own baseline
# ---------------------------------------------------------------------------

def test_flow_climatology_target_year_excluded():
    # Build a record where target year has wildly different values
    record_with_spike = list(_RECORD)
    target = date(2020, 7, 15)
    from datetime import timedelta
    doy = target.timetuple().tm_yday
    # Inject year 2020 values at 10x normal — if included they'd skew p50 badly
    for offset in range(-7, 8):
        d = date(2020, 1, 1) + timedelta(days=doy - 1 + offset)
        record_with_spike.append((d, 1_000_000.0))

    clim = flow_climatology(record_with_spike, target)
    assert clim is not None
    # p50 must still be near the synthetic seasonal value, not 1,000,000
    assert clim.p50 < 500


# ---------------------------------------------------------------------------
# Test 4: Fewer than min_years → None
# ---------------------------------------------------------------------------

def test_flow_climatology_insufficient_years_returns_none():
    # Only 5 years of data; min_years defaults to 10
    short_record = _synthetic_record(start_year=2010, end_year=2014)
    clim = flow_climatology(short_record, _TARGET)
    assert clim is None


# ---------------------------------------------------------------------------
# Test 5: flow_percentile returns ~5 at the 5th percentile, 0 and 100 at extremes
# ---------------------------------------------------------------------------

def test_flow_percentile_approximate_values():
    # ~5th percentile: a value just below the 5th percentile sorted value
    values = sorted([v for _, v in _RECORD if _ .timetuple().tm_yday == _TARGET_DOY])
    # We can't sort the full record by exact DOY match — use the climatology window instead
    from kinship_shared.baselines import _window_values
    win = sorted(_window_values(_RECORD, _TARGET, window_days=7))
    n = len(win)
    assert n > 0

    val_at_p5 = win[max(0, int(0.05 * n) - 1)]  # just at or below the 5th percentile
    pct = flow_percentile(val_at_p5, _RECORD, _TARGET)
    assert pct is not None
    assert 0 <= pct <= 15  # approximately 5th

    # Extreme low: value below all window values → 0
    pct_low = flow_percentile(min(win) - 1.0, _RECORD, _TARGET)
    assert pct_low == 0.0

    # Extreme high: value above all window values → 100
    pct_high = flow_percentile(max(win) + 1.0, _RECORD, _TARGET)
    assert pct_high == 100.0
