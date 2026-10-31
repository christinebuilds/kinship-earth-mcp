"""Tests for event synthesis pipeline (spec 013)."""

from datetime import datetime, timedelta, timezone

import pytest

from kinship_shared.event_classify import (
    BLOOM,
    DIE_OFF,
    DROUGHT_CASCADE,
    PHENOLOGICAL_SHIFT,
    EventPattern,
    synthesize_events,
)
from kinship_shared.schema import EcologicalAnomaly, Location


def _loc(lat: float = 37.5, lng: float = -122.0, name: str | None = None) -> Location:
    return Location(lat=lat, lng=lng, site_name=name)


def _anomaly(
    anomaly_type: str,
    severity: str = "warning",
    deviation_pct: float = -40.0,
    lat: float = 37.5,
    lng: float = -122.0,
    hours_ago: float = 1,
) -> EcologicalAnomaly:
    return EcologicalAnomaly(
        id=f"anomaly:{anomaly_type}:{lat}:{lng}",
        anomaly_type=anomaly_type,
        location=Location(lat=lat, lng=lng),
        detected_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        signal_value=60.0,
        baseline_value=100.0,
        deviation_pct=deviation_pct,
        severity=severity,
        description=f"Test {anomaly_type} anomaly",
        sources=["test"],
        confidence=0.85,
    )


def test_drought_cascade_detected():
    anomalies = [
        _anomaly("flow", severity="warning", deviation_pct=-45.0),
        _anomaly("temperature", severity="warning", deviation_pct=30.0),
    ]
    events = synthesize_events(anomalies, patterns=[DROUGHT_CASCADE])
    assert len(events) == 1
    assert events[0].event_type == "drought_cascade"


def test_die_off_detected():
    anomalies = [_anomaly("composition", severity="warning", deviation_pct=-60.0)]
    events = synthesize_events(anomalies, patterns=[DIE_OFF])
    assert len(events) == 1
    assert events[0].event_type == "die_off"


def test_phenological_shift_detected():
    anomalies = [_anomaly("phenological", severity="info", deviation_pct=25.0)]
    events = synthesize_events(anomalies, patterns=[PHENOLOGICAL_SHIFT])
    assert len(events) == 1
    assert events[0].event_type == "phenological_shift"


def test_bloom_detected():
    anomalies = [
        _anomaly("composition", severity="info", deviation_pct=80.0),
        _anomaly("temperature", severity="info", deviation_pct=20.0),
    ]
    events = synthesize_events(anomalies, patterns=[BLOOM])
    assert len(events) == 1
    assert events[0].event_type == "bloom"


def test_no_event_when_insufficient_anomalies():
    # Single temperature anomaly alone does NOT trigger drought_cascade (requires flow too)
    anomalies = [_anomaly("temperature", severity="warning", deviation_pct=30.0)]
    events = synthesize_events(anomalies, patterns=[DROUGHT_CASCADE])
    assert len(events) == 0


def test_events_sorted_by_severity():
    anomalies = [
        _anomaly("flow", severity="warning", deviation_pct=-45.0),
        _anomaly("temperature", severity="warning", deviation_pct=30.0),
        _anomaly("composition", severity="critical", deviation_pct=-80.0),
        _anomaly("phenological", severity="info", deviation_pct=20.0),
    ]
    events = synthesize_events(anomalies)
    assert len(events) >= 2
    sev_rank = {"emergency": 0, "critical": 1, "warning": 2, "info": 3}
    for i in range(len(events) - 1):
        assert sev_rank.get(events[i].severity, 4) <= sev_rank.get(events[i + 1].severity, 4)


def test_narrative_populated():
    anomalies = [
        _anomaly("flow", severity="warning", deviation_pct=-45.0),
        _anomaly("temperature", severity="warning", deviation_pct=30.0),
    ]
    events = synthesize_events(anomalies, patterns=[DROUGHT_CASCADE])
    assert len(events) == 1
    narrative = events[0].narrative
    assert len(narrative) > 20
    # Narrative should contain numeric deviation data
    assert any(c.isdigit() for c in narrative)


def test_spatial_filtering():
    center = _loc(lat=37.5, lng=-122.0)
    # Anomaly far away (>100km from center)
    far_anomaly = _anomaly("flow", lat=40.0, lng=-122.0)  # ~277km north
    near_anomaly = _anomaly("temperature", lat=37.5, lng=-122.0)
    events = synthesize_events(
        [far_anomaly, near_anomaly],
        patterns=[DROUGHT_CASCADE],
        location=center,
    )
    # Should not trigger: flow anomaly is outside 100km radius
    assert len(events) == 0


def test_temporal_filtering():
    # Old anomaly (35 days ago, outside the 30-day window)
    old_flow = _anomaly("flow", severity="warning", deviation_pct=-45.0, hours_ago=35 * 24)
    recent_temp = _anomaly("temperature", severity="warning", deviation_pct=30.0, hours_ago=1)
    events = synthesize_events([old_flow, recent_temp], patterns=[DROUGHT_CASCADE])
    # Old flow anomaly should be filtered out, no drought cascade
    assert len(events) == 0
