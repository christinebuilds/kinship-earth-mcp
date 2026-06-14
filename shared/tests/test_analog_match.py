"""Tests for the historical analog matching pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from kinship_shared.analog_match import (
    HistoricalAnalog,
    attach_analog_to_event,
    find_historical_analogs,
)
from kinship_shared.schema import EcologicalEvent, Location


def _make_event(
    event_type: str,
    lat: float = 42.0,
    lng: float = -121.8,
    severity: str = "warning",
) -> EcologicalEvent:
    return EcologicalEvent(
        id=f"event:{event_type}",
        event_type=event_type,
        location=Location(lat=lat, lng=lng),
        detected_at=datetime.now(timezone.utc),
        severity=severity,
        title=f"Test {event_type} event",
        narrative="A test event narrative.",
        sources=["era5", "usgs"],
        confidence=0.8,
    )


def test_find_analog_drought():
    event = _make_event("drought_cascade", lat=42.0, lng=-121.8)
    analogs = find_historical_analogs(event)
    assert len(analogs) > 0
    assert analogs[0].event_type == "drought_cascade"
    # Should find the Klamath Basin analog (close by)
    titles = [a.title for a in analogs]
    assert any("Klamath" in t for t in titles)


def test_find_analog_geographic_proximity():
    # Close to Klamath
    near_event = _make_event("drought_cascade", lat=42.0, lng=-121.8)
    # Far from Klamath (East Coast)
    far_event = _make_event("drought_cascade", lat=38.0, lng=-77.0)

    near_analogs = find_historical_analogs(near_event)
    far_analogs = find_historical_analogs(far_event)

    if near_analogs and far_analogs:
        # Near event's Klamath analog should score higher than far event's
        near_klamath = next((a for a in near_analogs if "Klamath" in a.title), None)
        far_klamath = next((a for a in far_analogs if "Klamath" in a.title), None)
        if near_klamath and far_klamath:
            assert near_klamath.similarity_score > far_klamath.similarity_score


def test_find_analog_no_match():
    # Migration event with no nearby analog in the knowledge base
    event = _make_event("migration", lat=0.0, lng=0.0)  # Equator/Africa, far from known analogs
    analogs = find_historical_analogs(event, max_distance_km=500.0)
    # Arctic tern analog is at lat=64, lng=-20 — far from equator
    assert len(analogs) == 0


def test_attach_analog_to_event():
    event = _make_event("drought_cascade", lat=42.0, lng=-121.8)
    analogs = find_historical_analogs(event)
    assert len(analogs) > 0
    result = attach_analog_to_event(event, analogs)
    assert result.historical_analog is not None
    assert "similarity" in result.historical_analog


def test_max_results_respected():
    event = _make_event("drought_cascade", lat=42.0, lng=-121.8)
    analogs = find_historical_analogs(event, max_results=1)
    assert len(analogs) <= 1


def test_no_cross_type_match():
    # die_off event should NOT match drought_cascade analogs
    event = _make_event("die_off", lat=42.0, lng=-121.8)
    analogs = find_historical_analogs(event)
    for a in analogs:
        assert a.event_type == "die_off"


def test_similarity_score_range():
    event = _make_event("drought_cascade", lat=42.0, lng=-121.8)
    analogs = find_historical_analogs(event)
    for a in analogs:
        assert 0.0 <= a.similarity_score <= 1.0
