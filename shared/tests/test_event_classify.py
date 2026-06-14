"""Tests for the event synthesis pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from kinship_shared.event_classify import (
    DEFAULT_PATTERNS,
    DROUGHT_CASCADE,
    DIE_OFF,
    PHENOLOGICAL_SHIFT,
    BLOOM,
    EventPattern,
    synthesize_events,
)
from kinship_shared.schema import EcologicalAnomaly, EcologicalEvent, Location


def _make_location(lat: float = 42.0, lng: float = -121.8) -> Location:
    return Location(lat=lat, lng=lng)


def _make_anomaly(
    anomaly_type: str,
    severity: str = "warning",
    deviation_pct: float = -40.0,
    lat: float = 42.0,
    lng: float = -121.8,
    detected_at: datetime | None = None,
) -> EcologicalAnomaly:
    baseline = 100.0
    return EcologicalAnomaly(
        id=f"anom:{anomaly_type}:{lat:.1f}_{lng:.1f}",
        anomaly_type=anomaly_type,
        location=Location(lat=lat, lng=lng),
        detected_at=detected_at or datetime.now(timezone.utc),
        severity=severity,
        deviation_pct=deviation_pct,
        signal_value=baseline + deviation_pct,
        baseline_value=baseline,
        sources=["era5", "usgs"],
        confidence=0.8,
        description=f"Test {anomaly_type} anomaly",
    )


def test_drought_cascade_detected():
    anomalies = [
        _make_anomaly("flow", severity="warning", deviation_pct=-45.0),
        _make_anomaly("temperature", severity="warning", deviation_pct=30.0),
    ]
    events = synthesize_events(anomalies, patterns=[DROUGHT_CASCADE])
    assert len(events) == 1
    assert events[0].event_type == "drought_cascade"


def test_die_off_detected():
    anomalies = [
        _make_anomaly("composition", severity="warning", deviation_pct=-60.0),
    ]
    events = synthesize_events(anomalies, patterns=[DIE_OFF])
    assert len(events) == 1
    assert events[0].event_type == "die_off"


def test_phenological_shift_detected():
    anomalies = [
        _make_anomaly("phenological", severity="info", deviation_pct=-25.0),
    ]
    events = synthesize_events(anomalies, patterns=[PHENOLOGICAL_SHIFT])
    assert len(events) == 1
    assert events[0].event_type == "phenological_shift"


def test_bloom_detected():
    anomalies = [
        _make_anomaly("composition", severity="info", deviation_pct=80.0),
        _make_anomaly("temperature", severity="info", deviation_pct=15.0),
    ]
    events = synthesize_events(anomalies, patterns=[BLOOM])
    assert len(events) == 1
    assert events[0].event_type == "bloom"


def test_no_event_when_insufficient_anomalies():
    # Only temperature — drought_cascade requires both flow AND temperature
    anomalies = [
        _make_anomaly("temperature", severity="warning", deviation_pct=30.0),
    ]
    events = synthesize_events(anomalies, patterns=[DROUGHT_CASCADE])
    assert len(events) == 0


def test_events_sorted_by_severity():
    anomalies = [
        _make_anomaly("flow", severity="warning", deviation_pct=-45.0),
        _make_anomaly("temperature", severity="critical", deviation_pct=60.0),
        _make_anomaly("composition", severity="warning", deviation_pct=-50.0),
        _make_anomaly("phenological", severity="info", deviation_pct=-20.0),
    ]
    events = synthesize_events(anomalies, patterns=DEFAULT_PATTERNS)
    if len(events) >= 2:
        severity_order = {"emergency": 0, "critical": 1, "warning": 2, "info": 3}
        ranks = [severity_order.get(e.severity, 4) for e in events]
        assert ranks == sorted(ranks), "Events should be sorted from most severe to least"


def test_narrative_populated():
    anomalies = [
        _make_anomaly("flow", severity="warning", deviation_pct=-45.0),
        _make_anomaly("temperature", severity="warning", deviation_pct=30.0),
    ]
    events = synthesize_events(anomalies, patterns=[DROUGHT_CASCADE])
    assert len(events) == 1
    assert len(events[0].narrative) > 20
    assert "%" in events[0].narrative


def test_spatial_filtering():
    center = _make_location(lat=42.0, lng=-121.8)
    # Anomaly far away (> 100km from center)
    far_anomaly = _make_anomaly("flow", lat=50.0, lng=-121.8)  # ~890 km away
    near_anomaly = _make_anomaly("temperature", lat=42.5, lng=-121.9)  # ~60 km

    events = synthesize_events([far_anomaly, near_anomaly], patterns=[DROUGHT_CASCADE], location=center)
    assert len(events) == 0  # far_anomaly excluded, so no drought (needs both flow + temp)


def test_temporal_filtering():
    from datetime import timedelta

    old_time = datetime.now(timezone.utc) - timedelta(days=60)
    old_anomaly = _make_anomaly("flow", detected_at=old_time)  # outside 30-day window
    recent_anomaly = _make_anomaly("temperature")  # within window

    events = synthesize_events([old_anomaly, recent_anomaly], patterns=[DROUGHT_CASCADE])
    assert len(events) == 0  # old anomaly excluded


def test_empty_anomalies_returns_empty():
    events = synthesize_events([])
    assert events == []
