"""Tests for anomaly detection pipeline."""

from datetime import datetime, timezone

from kinship_shared.anomaly_detect import (
    TemperatureDetector,
    FlowDetector,
    CompositionDetector,
    run_anomaly_detection,
    _classify_severity,
    _make_anomaly_id,
)
from kinship_shared.baselines import BaselineValues
from kinship_shared.schema import EcosystemState, Location


def _make_state(**kwargs):
    defaults = {
        "id": "test-site",
        "location": Location(lat=41.5, lng=-70.7),
        "timestamp": datetime(2023, 7, 15, tzinfo=timezone.utc),
        "sources_contributing": ["era5", "usgs-nwis"],
    }
    defaults.update(kwargs)
    return EcosystemState(**defaults)


def _make_baseline(**kwargs):
    defaults = {
        "location_id": "location:41.50_-70.70",
        "day_of_year": 196,
        "years_of_data": 5,
        "temp_mean_c": 20.0,
        "temp_std_c": 2.0,
    }
    defaults.update(kwargs)
    return BaselineValues(**defaults)


loc = Location(lat=41.5, lng=-70.7)


def test_temperature_detector_normal():
    state = _make_state(temp_mean_c=21.0)
    baseline = _make_baseline(temp_mean_c=20.0, temp_std_c=2.0)
    anomalies = TemperatureDetector().detect(location=loc, observations=[], baseline=baseline, state=state)
    assert len(anomalies) == 0


def test_temperature_detector_extreme_hot():
    state = _make_state(temp_mean_c=28.0)
    baseline = _make_baseline(temp_mean_c=20.0, temp_std_c=2.0)
    anomalies = TemperatureDetector().detect(location=loc, observations=[], baseline=baseline, state=state)
    assert len(anomalies) == 1
    assert anomalies[0].anomaly_type == "temperature"
    assert "above" in anomalies[0].description


def test_temperature_detector_extreme_cold():
    state = _make_state(temp_mean_c=14.0)
    baseline = _make_baseline(temp_mean_c=20.0, temp_std_c=2.0)
    anomalies = TemperatureDetector().detect(location=loc, observations=[], baseline=baseline, state=state)
    assert len(anomalies) == 1
    assert "below" in anomalies[0].description


def test_flow_detector_normal():
    state = _make_state(streamflow_cfs=100.0, streamflow_baseline=120.0)
    anomalies = FlowDetector().detect(location=loc, observations=[], state=state)
    assert len(anomalies) == 0


def test_flow_detector_drought():
    state = _make_state(streamflow_cfs=30.0, streamflow_baseline=100.0)
    anomalies = FlowDetector().detect(location=loc, observations=[], state=state)
    assert len(anomalies) == 1
    assert "drought" in anomalies[0].description


def test_flow_detector_flood():
    state = _make_state(streamflow_cfs=300.0, streamflow_baseline=100.0)
    anomalies = FlowDetector().detect(location=loc, observations=[], state=state)
    assert len(anomalies) == 1
    assert "flood" in anomalies[0].description


def test_composition_detector_normal():
    state = _make_state(species_richness=50, species_baseline=45)
    anomalies = CompositionDetector().detect(location=loc, observations=[], state=state)
    assert len(anomalies) == 0


def test_composition_detector_decline():
    state = _make_state(species_richness=20, species_baseline=50)
    anomalies = CompositionDetector().detect(location=loc, observations=[], state=state)
    assert len(anomalies) == 1
    assert "decrease" in anomalies[0].description


def test_run_anomaly_detection_aggregates():
    state = _make_state(temp_mean_c=28.0, streamflow_cfs=30.0, streamflow_baseline=100.0)
    baseline = _make_baseline(temp_mean_c=20.0, temp_std_c=2.0)
    anomalies = run_anomaly_detection(location=loc, baseline=baseline, state=state)
    assert len(anomalies) >= 2
    severities = [a.severity for a in anomalies]
    assert severities == sorted(severities, key=lambda s: {"critical": 0, "warning": 1, "info": 2}.get(s, 3))


def test_classify_severity_thresholds():
    assert _classify_severity(30) == "info"
    assert _classify_severity(75) == "warning"
    assert _classify_severity(150) == "critical"


def test_anomaly_id_format():
    dt = datetime(2023, 7, 15, tzinfo=timezone.utc)
    aid = _make_anomaly_id("temperature", 41.5, -70.7, dt)
    assert "anomaly:" in aid
    assert "2023-07-15" in aid
    assert "temperature" in aid


# ---------------------------------------------------------------------------
# Percentile-based flow anomalies (spec 015)
# ---------------------------------------------------------------------------

def test_flow_low_percentile_fires_warning():
    state = _make_state(
        streamflow_cfs=84.0, streamflow_baseline=310.0, streamflow_percentile=3,
        streamflow_baseline_kind="doy_record", streamflow_site_id="USGS-11467000",
        streamflow_record_years=62,
    )
    anomalies = FlowDetector().detect(location=loc, observations=[], state=state)
    assert len(anomalies) == 1
    a = anomalies[0]
    assert a.severity == "warning"
    assert "percentile" in a.description
    assert "62 years of record" in a.description and "USGS-11467000" in a.description
    assert a.deviation_pct < 0


def test_flow_median_percentile_is_quiet():
    state = _make_state(streamflow_cfs=300.0, streamflow_baseline=310.0, streamflow_percentile=50)
    assert FlowDetector().detect(location=loc, observations=[], state=state) == []


def test_flow_percentile_severity_bands():
    def sev(p):
        s = _make_state(streamflow_cfs=10.0, streamflow_baseline=100.0, streamflow_percentile=p)
        return FlowDetector().detect(location=loc, observations=[], state=s)[0].severity
    assert sev(1) == "critical" and sev(99) == "critical"
    assert sev(5) == "warning" and sev(96) == "warning"
    assert sev(10) == "info" and sev(90) == "info"


def test_flow_without_percentile_uses_percent_of_mean():
    state = _make_state(streamflow_cfs=200.0, streamflow_baseline=100.0)
    anomalies = FlowDetector().detect(location=loc, observations=[], state=state)
    assert len(anomalies) == 1
    assert "above normal" in anomalies[0].description
