"""
Movebank Adapter — animal tracking and GPS telemetry data.

Movebank is a free, online database of animal tracking data hosted
by the Max Planck Institute of Animal Behavior. It provides access
to GPS, Argos, radio, and accelerometer data from tagged animals.

API base: https://www.movebank.org/movebank/service/direct-read
Auth: Requires free account (username + password for basic auth)
      Set MOVEBANK_USER and MOVEBANK_PASSWORD env vars.
Coverage: Global, 6,000+ studies, emphasis on migration
Data: GPS locations, timestamps, sensor readings, study metadata
Temporal resolution: Minutes to hours (depends on tag configuration)
Quality tier: 1 (calibrated instrument data)

This adapter bridges animal movement data into the Kinship Earth
schema, enabling cross-source queries like "what birds are migrating
through this watershed while streamflow is anomalously low?"
"""

from __future__ import annotations

import csv
import io
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from kinship_shared import (
    AdapterCapabilities,
    EcologicalAdapter,
    EcologicalObservation,
    Location,
    Provenance,
    Quality,
    SearchParams,
    TaxonInfo,
)
from kinship_shared.retry import http_get_with_retry

logger = logging.getLogger(__name__)

MOVEBANK_API_BASE = "https://www.movebank.org/movebank/service/direct-read"


class MovebankAdapter(EcologicalAdapter):
    """Adapter for Movebank animal tracking data."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
    ):
        self._username = username or os.environ.get("MOVEBANK_USER", "")
        self._password = password or os.environ.get("MOVEBANK_PASSWORD", "")
        self._auth = (self._username, self._password) if self._username else None

    @property
    def id(self) -> str:
        return "movebank"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_id="movebank",
            name="Movebank",
            description="Animal tracking and GPS telemetry data from the Max Planck Institute. 6,000+ studies covering migration, movement ecology, and habitat use.",
            modalities=["movement"],
            supports_location_search=True,
            supports_taxon_search=True,
            supports_date_range=True,
            geographic_coverage="global",
            temporal_coverage_start="2000-01-01",
            update_frequency="real-time",
            quality_tier=1,
            requires_auth=True,
            rate_limit_per_minute=60,
            license="varies-by-study",
            homepage_url="https://www.movebank.org",
        )

    async def search(self, params: SearchParams) -> list[EcologicalObservation]:
        """Search Movebank for animal tracking data.

        Strategy:
        1. Search for studies matching the taxon and/or bounding box
        2. For matching studies, fetch individual locations (GPS points)
        3. Convert to EcologicalObservation with modality='movement'
        """
        if not self._auth:
            logger.warning("Movebank credentials not set (MOVEBANK_USER, MOVEBANK_PASSWORD)")
            return []

        observations: list[EcologicalObservation] = []

        try:
            # Step 1: Find relevant studies
            study_params: dict = {"entity_type": "study"}
            if params.taxon:
                study_params["taxon_name"] = params.taxon

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    MOVEBANK_API_BASE,
                    params=study_params,
                    auth=self._auth,
                )
                if resp.status_code != 200:
                    logger.warning("Movebank study search failed: %s", resp.status_code)
                    return []

                studies = self._parse_csv(resp.text)

                if not studies:
                    return []

                # Step 2: For top studies, fetch recent locations
                for study in studies[:3]:  # Limit to 3 studies to stay within rate limits
                    study_id = study.get("id", "")
                    if not study_id:
                        continue

                    location_params: dict = {
                        "entity_type": "event",
                        "study_id": study_id,
                        "attributes": "individual_local_identifier,timestamp,location_long,location_lat,ground_speed,heading",
                        "max_events_per_individual": str(min(params.limit, 50)),
                    }
                    if params.start_date:
                        location_params["timestamp_start"] = params.start_date
                    if params.end_date:
                        location_params["timestamp_end"] = params.end_date

                    try:
                        loc_resp = await client.get(
                            MOVEBANK_API_BASE,
                            params=location_params,
                            auth=self._auth,
                        )
                        if loc_resp.status_code != 200:
                            continue

                        events = self._parse_csv(loc_resp.text)

                        for event in events:
                            obs = self._event_to_observation(event, study)
                            if obs is None:
                                continue

                            # Filter by location if specified
                            if params.lat is not None and params.lng is not None:
                                from math import radians, sin, cos, sqrt, atan2
                                R = 6371
                                dlat = radians(obs.location.lat - params.lat)
                                dlon = radians(obs.location.lng - params.lng)
                                a = sin(dlat/2)**2 + cos(radians(params.lat)) * cos(radians(obs.location.lat)) * sin(dlon/2)**2
                                dist = R * 2 * atan2(sqrt(a), sqrt(1-a))
                                radius = params.radius_km or 200
                                if dist > radius:
                                    continue

                            observations.append(obs)

                            if len(observations) >= params.limit:
                                return observations

                    except Exception as e:
                        logger.warning("Failed to fetch events for study %s: %s", study_id, e)
                        continue

        except Exception as e:
            logger.warning("Movebank search failed: %s", e)

        return observations

    async def get_by_id(self, source_id: str) -> Optional[EcologicalObservation]:
        """Fetch a specific tracking event by ID. Not directly supported by Movebank."""
        return None

    def _parse_csv(self, csv_text: str) -> list[dict]:
        """Parse Movebank's CSV response into list of dicts."""
        if not csv_text.strip() or csv_text.startswith("No data"):
            return []
        reader = csv.DictReader(io.StringIO(csv_text))
        return list(reader)

    def _event_to_observation(
        self, event: dict, study: dict
    ) -> EcologicalObservation | None:
        """Convert a Movebank event record to EcologicalObservation."""
        try:
            lat = float(event.get("location_lat", ""))
            lng = float(event.get("location_long", ""))
        except (ValueError, TypeError):
            return None

        try:
            timestamp = datetime.fromisoformat(
                event.get("timestamp", "").replace(" ", "T")
            )
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)

        taxon_name = study.get("taxon_ids", "") or study.get("main_location_long", "")
        individual = event.get("individual_local_identifier", "unknown")
        study_name = study.get("name", "")
        study_id = study.get("id", "")

        value: dict = {"individual_id": individual}
        if event.get("ground_speed"):
            try:
                value["speed_m_s"] = float(event["ground_speed"])
            except (ValueError, TypeError):
                pass
        if event.get("heading"):
            try:
                value["heading_deg"] = float(event["heading"])
            except (ValueError, TypeError):
                pass

        obs_id = f"movebank:{study_id}:{individual}:{timestamp.isoformat()}"

        return EcologicalObservation(
            id=obs_id,
            modality="movement",
            taxon=TaxonInfo(
                scientific_name=taxon_name or "Unknown",
                common_name=study_name or None,
            ),
            location=Location(lat=lat, lng=lng),
            observed_at=timestamp,
            value=value,
            unit="GPS fix",
            quality=Quality(
                tier=1,
                grade="research",
                confidence=0.95,
            ),
            provenance=Provenance(
                source_api="movebank",
                source_id=obs_id,
                original_url=f"https://www.movebank.org/cms/webapp?gwt_fragment=page=studies,path=study{study_id}",
                license="varies-by-study",
                attribution=f"Movebank study: {study_name}",
                collection_method="GPS telemetry",
                sensor_id=individual,
            ),
            temporal_resolution="event-driven",
        )
