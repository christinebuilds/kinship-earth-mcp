"""
Event synthesis — correlates multiple anomalies into classified events.

A single anomaly is a signal. An event is a story: "low streamflow +
high temperature + species richness decline = drought cascade."

Each EventPattern defines a rule: which anomaly types must co-occur,
how close in space/time, and what event type to emit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel, Field

from .schema import (
    EcologicalAnomaly,
    EcologicalEvent,
    EcologicalEventType,
    Location,
)

logger = logging.getLogger(__name__)


class EventPattern(BaseModel):
    """A rule for synthesizing anomalies into an event.

    When all required anomaly types are present within the spatial and
    temporal windows, the event fires.
    """

    event_type: EcologicalEventType
    required_anomaly_types: list[str] = Field(
        description="Anomaly types that must all be present"
    )
    min_anomalies: int = Field(
        default=2,
        description="Minimum number of anomalies required to trigger"
    )
    time_window_days: int = Field(
        default=30,
        description="Anomalies must be within this many days of each other"
    )
    spatial_radius_km: float = Field(
        default=100.0,
        description="Anomalies must be within this radius to correlate"
    )
    title_template: str = Field(
        description="Template for event title, e.g. 'Drought cascade at {location}'"
    )
    narrative_template: str = Field(
        description="Template for event narrative"
    )
    min_severity: str = Field(
        default="warning",
        description="Minimum severity of contributing anomalies"
    )


# ---------------------------------------------------------------------------
# Built-in event patterns
# ---------------------------------------------------------------------------

DROUGHT_CASCADE = EventPattern(
    event_type="drought_cascade",
    required_anomaly_types=["flow", "temperature"],
    min_anomalies=2,
    time_window_days=30,
    spatial_radius_km=100.0,
    title_template="Drought cascade at {location}",
    narrative_template=(
        "Multiple drought signals detected: streamflow is {flow_dev}% below normal "
        "while temperature is {temp_dev}% above normal. This combination suggests "
        "compound drought stress on aquatic and riparian ecosystems."
    ),
    min_severity="warning",
)

DIE_OFF = EventPattern(
    event_type="die_off",
    required_anomaly_types=["composition"],
    min_anomalies=1,
    time_window_days=14,
    spatial_radius_km=50.0,
    title_template="Potential die-off event at {location}",
    narrative_template=(
        "Species richness has dropped {comp_dev}% from baseline levels. "
        "This rapid decline in biodiversity may indicate a die-off event. "
        "Immediate field verification recommended."
    ),
    min_severity="warning",
)

PHENOLOGICAL_SHIFT = EventPattern(
    event_type="phenological_shift",
    required_anomaly_types=["phenological"],
    min_anomalies=1,
    time_window_days=60,
    spatial_radius_km=200.0,
    title_template="Phenological shift detected at {location}",
    narrative_template=(
        "Species activity patterns are {phen_dev}% different from expected "
        "for this time of year. This may indicate shifting seasonal timing "
        "due to climate change or habitat alteration."
    ),
    min_severity="info",
)

BLOOM = EventPattern(
    event_type="bloom",
    required_anomaly_types=["composition", "temperature"],
    min_anomalies=2,
    time_window_days=14,
    spatial_radius_km=50.0,
    title_template="Possible bloom event at {location}",
    narrative_template=(
        "Species composition increase of {comp_dev}% combined with "
        "temperature anomaly of {temp_dev}% suggests a bloom event "
        "(algal, insect emergence, or similar rapid population increase)."
    ),
    min_severity="info",
)

DEFAULT_PATTERNS: list[EventPattern] = [
    DROUGHT_CASCADE,
    DIE_OFF,
    PHENOLOGICAL_SHIFT,
    BLOOM,
]


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    import math
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _anomalies_in_window(
    anomalies: list[EcologicalAnomaly],
    center: Location,
    radius_km: float,
    time_window_days: int,
    reference_time: datetime,
) -> list[EcologicalAnomaly]:
    """Filter anomalies to those within spatial and temporal window."""
    cutoff = reference_time - timedelta(days=time_window_days)
    return [
        a for a in anomalies
        if (
            a.detected_at >= cutoff
            and _haversine_km(center.lat, center.lng, a.location.lat, a.location.lng) <= radius_km
        )
    ]


# ---------------------------------------------------------------------------
# Event synthesis
# ---------------------------------------------------------------------------

def _format_narrative(template: str, anomalies: list[EcologicalAnomaly]) -> str:
    """Fill in narrative template with anomaly deviation values."""
    values = {}
    for a in anomalies:
        prefix = a.anomaly_type[:4]  # e.g. "temp", "flow", "comp", "phen"
        values[f"{prefix}_dev"] = f"{abs(a.deviation_pct):.0f}"
    try:
        return template.format(**values, location="the target area")
    except KeyError:
        result = template
        for k, v in values.items():
            result = result.replace(f"{{{k}}}", v)
        return result.replace("{location}", "the target area")


def synthesize_events(
    anomalies: list[EcologicalAnomaly],
    patterns: list[EventPattern] | None = None,
    location: Location | None = None,
) -> list[EcologicalEvent]:
    """Synthesize anomalies into classified ecological events.

    For each pattern, checks if the required anomaly types are present
    within the spatial/temporal window. If so, creates an EcologicalEvent.

    Args:
        anomalies: All detected anomalies (may span locations and times).
        patterns: Event patterns to check. Defaults to DEFAULT_PATTERNS.
        location: Optional center point for spatial filtering. If None, uses
                  the location of the first anomaly.

    Returns:
        List of synthesized events, sorted by severity.
    """
    if not anomalies:
        return []

    patterns = patterns or DEFAULT_PATTERNS
    center = location or anomalies[0].location
    now = datetime.now(timezone.utc)
    events: list[EcologicalEvent] = []

    severity_order = {"info": 0, "warning": 1, "critical": 2}

    for pattern in patterns:
        windowed = _anomalies_in_window(
            anomalies, center, pattern.spatial_radius_km,
            pattern.time_window_days, now,
        )

        min_sev = severity_order.get(pattern.min_severity, 0)
        windowed = [a for a in windowed if severity_order.get(a.severity, 0) >= min_sev]

        present_types = {a.anomaly_type for a in windowed}
        required = set(pattern.required_anomaly_types)
        if not required.issubset(present_types):
            continue

        matching = [a for a in windowed if a.anomaly_type in required]
        if len(matching) < pattern.min_anomalies:
            continue

        max_severity = max(
            (severity_order.get(a.severity, 0) for a in matching),
            default=0,
        )
        event_severity_map = {0: "info", 1: "warning", 2: "critical"}
        event_severity = event_severity_map.get(max_severity, "info")

        timestamps = [a.detected_at for a in matching]
        duration_days = max(1, (max(timestamps) - min(timestamps)).days) if len(timestamps) > 1 else 1

        narrative = _format_narrative(pattern.narrative_template, matching)

        all_sources = list(set(s for a in matching for s in a.sources))

        confidence = sum(a.confidence for a in matching) / len(matching)

        event_id = f"event:{pattern.event_type}:{center.lat:.2f}_{center.lng:.2f}:{now.strftime('%Y-%m-%d')}"

        location_name = center.site_name or f"{center.lat:.2f}, {center.lng:.2f}"
        title = pattern.title_template.replace("{location}", location_name)

        events.append(EcologicalEvent(
            id=event_id,
            event_type=pattern.event_type,
            location=center,
            detected_at=now,
            duration_days=duration_days,
            severity=event_severity,
            title=title,
            narrative=narrative,
            anomalies=[a.id for a in matching],
            sources=all_sources,
            confidence=round(confidence, 2),
        ))

    sev_rank = {"emergency": 0, "critical": 1, "warning": 2, "info": 3}
    events.sort(key=lambda e: (sev_rank.get(e.severity, 4), -e.confidence))

    return events
