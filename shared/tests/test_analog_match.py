"""Tests for historical analog matching (spec 013)."""

from datetime import datetime, timezone

import pytest

from kinship_shared.analog_match import (
    HistoricalAnalog,
    attach_analog_to_event,
    find_historical_analogs,
)
from kinship_shared.schema import EcologicalEvent, Location


def _event(
    event_type: str,
    lat: float = 37.5,
    lng: float = -122.0,
    severity: str = "warning",
) -> EcologicalEvent:
    return EcologicalEvent(
        id=f"event:{event_type}:test",
        event_type=event_type,
        location=Location(lat=lat, lng=lng),
        detected_at=datetime.now(timezone.utc),
        severity=severity,
        title=f"Test {event_type}",
        narrative="Test narrative.",
        confidence=0.8,
    )


def test_find_analog_drought():
    event = _event("drought_cascade", lat=42.0, lng=-122.0)
    analogs = find_historical_analogs(event)
    assert len(analogs) > 0
    # Klamath Basin is at 42.0, -121.8 — very close
    titles = [a.title for a in analogs]
    assert any("Klamath" in t for t in titles)


def test_find_analog_geographic_proximity():
    # Event near Klamath Basin should score Klamath higher than Colorado River
    event_klamath = _event("drought_cascade", lat=42.0, lng=-121.8)
    analogs = find_historical_analogs(event_klamath)
    assert len(analogs) >= 2
    # First result should have higher similarity than second
    assert analogs[0].similarity_score >= analogs[1].similarity_score
    # Klamath should be first (closest)
    assert "Klamath" in analogs[0].title


def test_find_analog_no_match():
    # Migration event with location far from the North Atlantic analog
    event = _event("migration", lat=-33.0, lng=151.0)  # Sydney, Australia
    analogs = find_historical_analogs(event, max_distance_km=500.0)
    assert len(analogs) == 0


def test_attach_analog_to_event():
    event = _event("drought_cascade", lat=42.0, lng=-122.0)
    analogs = find_historical_analogs(event)
    result = attach_analog_to_event(event, analogs)
    assert result.historical_analog is not None
    assert "Klamath" in result.historical_analog
    assert "similarity" in result.historical_analog


def test_max_results_respected():
    event = _event("drought_cascade", lat=40.0, lng=-115.0)
    analogs = find_historical_analogs(event, max_results=1)
    assert len(analogs) <= 1
