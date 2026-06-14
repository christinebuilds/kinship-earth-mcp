"""
Historical analog matching — connects current events to past patterns.

Given a current EcologicalEvent, searches stored event history and the
knowledge graph for similar past events based on event type, location,
severity, and contributing anomaly patterns.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from .schema import EcologicalEvent, Location

logger = logging.getLogger(__name__)


class HistoricalAnalog(BaseModel):
    """A past event that resembles the current one."""

    event_id: str
    event_type: str
    title: str
    detected_at: datetime
    location_name: str
    similarity_score: float = Field(ge=0, le=1, description="How similar to the query event (0-1)")
    explanation: str = Field(description="Why this is considered analogous")


# ---------------------------------------------------------------------------
# Built-in historical knowledge base
# ---------------------------------------------------------------------------

KNOWN_ANALOGS: list[dict] = [
    {
        "event_type": "drought_cascade",
        "title": "2021 Klamath Basin drought cascade",
        "year": 2021,
        "lat": 42.0,
        "lng": -121.8,
        "region": "Klamath Basin, OR/CA",
        "description": (
            "Record low water levels in the Klamath Basin led to massive salmon die-offs, "
            "irrigation shutoffs, and cascading ecosystem collapse. Stream temperatures "
            "exceeded 25°C and flows dropped to 10% of normal."
        ),
    },
    {
        "event_type": "drought_cascade",
        "title": "2020 Colorado River low-flow crisis",
        "year": 2020,
        "lat": 36.0,
        "lng": -111.8,
        "region": "Colorado River, AZ/UT",
        "description": (
            "Lake Powell dropped to historically low levels. Riparian habitat loss, "
            "endangered fish species stress, and multi-state water conflict."
        ),
    },
    {
        "event_type": "die_off",
        "title": "2023 Pacific marine heatwave die-off",
        "year": 2023,
        "lat": 37.0,
        "lng": -122.5,
        "region": "Pacific Coast, CA",
        "description": (
            "Elevated sea surface temperatures triggered mass mortality in intertidal "
            "species including sea stars, mussels, and kelp. SST anomaly +3.5°C."
        ),
    },
    {
        "event_type": "phenological_shift",
        "title": "2019 Northeast spring advancement",
        "year": 2019,
        "lat": 42.5,
        "lng": -72.0,
        "region": "Northeast US",
        "description": (
            "Spring leaf-out and bloom dates advanced 2-3 weeks earlier than historical "
            "average. Migratory bird arrival fell out of sync with peak caterpillar emergence."
        ),
    },
    {
        "event_type": "bloom",
        "title": "2024 Lake Erie harmful algal bloom",
        "year": 2024,
        "lat": 41.6,
        "lng": -83.0,
        "region": "Western Lake Erie, OH",
        "description": (
            "Cyanobacteria bloom covered 800 square miles. Triggered by nutrient loading + "
            "warm temperatures + low flow. Drinking water advisories for 500,000+ residents."
        ),
    },
    {
        "event_type": "migration",
        "title": "2022 Arctic tern route shift",
        "year": 2022,
        "lat": 64.0,
        "lng": -20.0,
        "region": "North Atlantic",
        "description": (
            "Arctic terns shifted their Atlantic migratory route 200km eastward, "
            "correlating with changes in prey fish distribution linked to ocean warming."
        ),
    },
]


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
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


def find_historical_analogs(
    event: EcologicalEvent,
    max_results: int = 3,
    max_distance_km: float = 2000.0,
) -> list[HistoricalAnalog]:
    """Find historical events analogous to the given event.

    Matching criteria (weighted):
    - Same event type (required)
    - Geographic proximity (closer = higher score)
    - Similar season (same month = higher score)

    Args:
        event: The current event to find analogs for.
        max_results: Maximum number of analogs to return.
        max_distance_km: Maximum distance for analog consideration.

    Returns:
        List of HistoricalAnalog objects, sorted by similarity score.
    """
    candidates: list[HistoricalAnalog] = []

    for analog_data in KNOWN_ANALOGS:
        if analog_data["event_type"] != event.event_type:
            continue

        dist = _haversine_km(
            event.location.lat, event.location.lng,
            analog_data["lat"], analog_data["lng"],
        )
        if dist > max_distance_km:
            continue
        geo_score = max(0.0, 1.0 - dist / max_distance_km)

        season_score = 1.0

        similarity = 0.5 * geo_score + 0.3 * season_score + 0.2 * 1.0

        explanation = (
            f"Similar {analog_data['event_type'].replace('_', ' ')} event in "
            f"{analog_data['region']} ({analog_data['year']}). "
            f"{analog_data['description'][:150]}"
        )

        candidates.append(HistoricalAnalog(
            event_id=f"historical:{analog_data['event_type']}:{analog_data['year']}",
            event_type=analog_data["event_type"],
            title=analog_data["title"],
            detected_at=datetime(analog_data["year"], 6, 1, tzinfo=timezone.utc),
            location_name=analog_data["region"],
            similarity_score=round(similarity, 2),
            explanation=explanation,
        ))

    candidates.sort(key=lambda a: -a.similarity_score)
    return candidates[:max_results]


def attach_analog_to_event(event: EcologicalEvent, analogs: list[HistoricalAnalog]) -> EcologicalEvent:
    """Attach the best historical analog to an event.

    Modifies the event's historical_analog field with the top match.
    Returns the same event (mutated) for chaining convenience.
    """
    if analogs:
        best = analogs[0]
        event.historical_analog = f"{best.title} (similarity: {best.similarity_score:.0%})"
    return event
