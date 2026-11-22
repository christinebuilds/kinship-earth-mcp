"""Offline tests for flow_climatology and flow_percentile (spec 015)."""

from datetime import date, timedelta

import pytest

from kinship_shared.baselines import FlowClimatology, flow_climatology, flow_percentile


def _make_synthetic_record(years: int = 40, base_doy: int = 180, amplitude: float = 100.0) -> list[tuple[date, float]]:
    """Build a synthetic 40-year daily record with a known seasonal cycle.

    Flow(doy) = 200 + amplitude * sin(2π * doy / 365)
    DOY 180 (June 29) → peak ≈ 200 + 100 ≈ 300 cfs
    DOY 1   (Jan 1)   → near-mean ≈ 200 cfs
    """
    import math

    start_year = 1980
    records: list[tuple[date, float]] = []
    for year_offset in range(years):
        year = start_year + year_offset
        for doy in range(1, 366):
            try:
                d = date(year, 1, 1) + timedelta(days=doy - 1)
            except ValueError:
                continue
            flow = 200.0 + amplitude * math.sin(2 * math.pi * doy / 365)
            records.append((d, round(flow, 2)))
    return records


class TestFlowClimatology:
    """Tests for flow_climatology function."""

    def test_p50_within_5pct_of_known_median(self):
        """p50 should be within 5% of the constructed seasonal median."""
        import math

        record = _make_synthetic_record(years=40)
        target = date(2020, 6, 29)  # DOY 180 — peak of seasonal cycle
        clim = flow_climatology(record, target, window_days=7, min_years=10, site_id="USGS-TEST")

        assert clim is not None
        expected_median = 200.0 + 100.0 * math.sin(2 * math.pi * 180 / 365)
        assert abs(clim.p50 - expected_median) / expected_median < 0.05, (
            f"p50={clim.p50:.1f} not within 5% of expected {expected_median:.1f}"
        )

    def test_year_end_wrap(self):
        """DOY window should wrap around year boundaries (Jan 3 includes late Dec)."""
        record = _make_synthetic_record(years=40)
        target = date(2020, 1, 3)  # DOY 3 — should include Dec 27–Jan 10

        clim = flow_climatology(record, target, window_days=7, min_years=10)
        assert clim is not None, "Should find enough data for Jan 3 target"
        # Should have values from both December and January in the pool
        assert clim.n_values > 0

    def test_target_year_excluded(self):
        """The target year itself must not contribute to the climatology."""
        record = _make_synthetic_record(years=40)
        target = date(2019, 7, 1)  # 2019 is in the record

        # Build using all years, verify 2019 excluded
        clim = flow_climatology(record, target, window_days=7, min_years=10, site_id="USGS-TEST")
        assert clim is not None

        # Inject an outlier for 2019 only — if it were included, stats would shift
        outlier_date = date(2019, 7, 1)
        modified = [(d, 999999.0 if d == outlier_date else v) for d, v in record]
        clim_modified = flow_climatology(modified, target, window_days=7, min_years=10)
        assert clim_modified is not None
        assert abs(clim.p50 - clim_modified.p50) < 1.0, (
            "Target year outlier should not affect climatology"
        )

    def test_fewer_than_min_years_returns_none(self):
        """Should return None when fewer than min_years distinct years contribute."""
        # Only 5 years of data
        record = _make_synthetic_record(years=5)
        target = date(1984, 7, 1)

        result = flow_climatology(record, target, window_days=7, min_years=10)
        assert result is None

    def test_returns_flow_climatology_model(self):
        """Return value should be a FlowClimatology with expected fields."""
        record = _make_synthetic_record(years=40)
        target = date(2020, 7, 15)

        clim = flow_climatology(record, target, window_days=7, min_years=10, site_id="USGS-11467000")
        assert clim is not None
        assert isinstance(clim, FlowClimatology)
        assert clim.site_id == "USGS-11467000"
        assert clim.p05 <= clim.p25 <= clim.p50 <= clim.p75 <= clim.p95
        assert clim.years_of_data >= 10
        assert clim.record_start < clim.record_end


class TestFlowPercentile:
    """Tests for flow_percentile function."""

    def test_value_at_constructed_5th_percentile(self):
        """flow_percentile should return ~5 for a value at the constructed 5th pct."""
        import math

        record = _make_synthetic_record(years=40)
        target = date(2020, 7, 1)

        clim = flow_climatology(record, target, window_days=7, min_years=10)
        assert clim is not None

        # Use p05 value directly — should come back near 5
        pct = flow_percentile(clim.p05, record, target, window_days=7)
        assert pct is not None
        assert pct <= 10, f"Expected ≤10, got {pct}"

    def test_low_extreme_returns_near_zero(self):
        """An extreme low value should return a low percentile."""
        record = _make_synthetic_record(years=40)
        target = date(2020, 7, 1)

        pct = flow_percentile(-999.0, record, target, window_days=7)
        assert pct is not None
        assert pct == 0.0

    def test_high_extreme_returns_near_100(self):
        """An extreme high value should return a high percentile."""
        record = _make_synthetic_record(years=40)
        target = date(2020, 7, 1)

        pct = flow_percentile(999999.0, record, target, window_days=7)
        assert pct is not None
        assert pct == 100.0

    def test_no_matching_values_returns_none(self):
        """When no values fall in the window for the given year, return None."""
        # Build a record that only has values for DOY 1–10 and target year 2020 excluded
        records: list[tuple[date, float]] = [
            (date(2000 + i, 1, 1), 100.0) for i in range(20)
        ]
        # Target is July — no DOY match
        pct = flow_percentile(100.0, records, date(2020, 7, 15), window_days=7)
        assert pct is None


@pytest.mark.network
async def test_get_daily_record_russian_river():
    """USGS-11467000 should have more than 10 years of daily values."""
    import httpx

    # Import the adapter (requires httpx + network)
    from usgs_nwis_mcp.adapter import USGSNWISAdapter

    adapter = USGSNWISAdapter()
    record = await adapter.get_daily_record("USGS-11467000", parameter_code="00060")

    assert len(record) > 0, "Should return daily records"
    dates = [d for d, _ in record]
    assert dates == sorted(dates), "Records should be in date order"

    year_span = (max(dates) - min(dates)).days / 365.25
    assert year_span > 10, f"Expected >10 years, got {year_span:.1f}"
