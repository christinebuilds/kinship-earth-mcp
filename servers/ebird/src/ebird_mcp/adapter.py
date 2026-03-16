"""
eBird Adapter — Cornell Lab of Ornithology bird observation data.

eBird API 2.0: https://documenter.getpostman.com/view/664302/S1ENwy59
Auth: API key required (free from https://ebird.org/api/keygen)
Coverage: Global, ~1.5 billion observations, near real-time
Rate limit: Not officially documented; be conservative (100 req/min)

eBird includes observations from the Merlin bird ID app (audio-based
species identification). Merlin observations flow into eBird automatically
and are subject to the same review process.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from math import radians, sin, cos, sqrt, atan2
from typing import Optional

from kinship_shared import (
    EcologicalAdapter,
    AdapterCapabilities,
    EcologicalObservation,
    Location,
    TaxonInfo,
    Quality,
    Provenance,
    SearchParams,
)
import httpx

from kinship_shared.retry import http_get_with_retry

# eBird API 2.0 base URL
BASE_URL = "https://api.ebird.org/v2"


class EBirdAdapter(EcologicalAdapter):
    """Adapter for the eBird API (Cornell Lab of Ornithology)."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("EBIRD_API_KEY")

    @property
    def id(self) -> str:
        return "ebird"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_id="ebird",
            name="eBird (Cornell Lab of Ornithology)",
            description=(
                "The world's largest biodiversity citizen science database, with "
                "over 1.5 billion bird observations contributed by birders worldwide. "
                "Includes observations from the Merlin bird ID app (audio + photo AI). "
                "Data undergoes automated and expert review for quality assurance."
            ),
            modalities=["occurrence"],
            supports_location_search=True,
            supports_taxon_search=True,
            supports_date_range=True,  # limited — recent observations only via API
            supports_site_search=False,
            geographic_coverage="Global",
            temporal_coverage_start="2002-01-01",
            update_frequency="real-time",
            quality_tier=1,  # Research grade — expert-reviewed
            requires_auth=True,
            rate_limit_per_minute=100,
            license="CC0 for personal/research use; see eBird Terms of Use",
            homepage_url="https://ebird.org",
        )

    async def search(self, params: SearchParams) -> list[EcologicalObservation]:
        """
        Search eBird for recent bird observations.

        eBird API supports:
        - Recent observations near a location (lat/lng + radius)
        - Recent observations of a species near a location
        - Notable/rare sightings near a location

        Limitations:
        - API only returns recent observations (last 1-30 days)
        - Historical data requires the eBird data download (not API)
        - No arbitrary date range queries via API
        """
        if not self._api_key:
            return []  # Cannot query without API key

        observations = []

        if params.lat is not None and params.lng is not None:
            # Search for recent observations near a location
            dist_km = min(params.radius_km or 25, 50)  # eBird max is 50km

            if params.taxon:
                # DECISION: Need to map scientific name → eBird species code
                # eBird uses 6-letter species codes (e.g., "norcar" for Northern Cardinal)
                # For now, use the taxonomy endpoint to resolve, or fall back to
                # recent observations and filter client-side
                endpoint = f"{BASE_URL}/data/obs/geo/recent"
            else:
                endpoint = f"{BASE_URL}/data/obs/geo/recent"

            api_params = {
                "lat": params.lat,
                "lng": params.lng,
                "dist": dist_km,
                "maxResults": min(params.limit, 100),  # eBird max is 100 per page
                "includeProvisional": "true",
                "back": 14,  # days back (1-30)
            }

            headers = {"X-eBirdApiToken": self._api_key}
            async with httpx.AsyncClient(timeout=30, headers=headers) as client:
                resp = await http_get_with_retry(
                    client,
                    endpoint,
                    params=api_params,
                )
                resp.raise_for_status()
                data = resp.json()

            if isinstance(data, list):
                for record in data:
                    obs = self._record_to_observation(record)
                    if obs is not None:
                        # Client-side taxon filter if specified
                        if params.taxon and obs.taxon:
                            if params.taxon.lower() not in obs.taxon.scientific_name.lower():
                                continue
                        observations.append(obs)

        return observations[:params.limit]

    async def get_by_id(self, source_id: str) -> Optional[EcologicalObservation]:
        """eBird API does not support fetching individual observations by ID."""
        # DECISION: eBird's API doesn't have a get-by-ID endpoint for observations.
        # Could potentially use the checklist endpoint if we have the checklist ID.
        return None

    def _record_to_observation(self, record: dict) -> Optional[EcologicalObservation]:
        """
        Convert an eBird API observation record to EcologicalObservation.

        eBird API response fields (from /data/obs/geo/recent):
        - speciesCode: 6-letter species code (e.g., "norcar")
        - comName: common name (e.g., "Northern Cardinal")
        - sciName: scientific name (e.g., "Cardinalis cardinalis")
        - locId: location ID (e.g., "L123456")
        - locName: location name
        - lat: latitude
        - lng: longitude
        - obsDt: observation date (YYYY-MM-DD HH:MM or YYYY-MM-DD)
        - howMany: count (may be null for presence-only)
        - subId: checklist/submission ID
        - obsValid: whether observation has been validated
        - obsReviewed: whether observation has been reviewed
        - locationPrivate: whether location is private
        """
        try:
            lat = record.get("lat")
            lng = record.get("lng")
            sci_name = record.get("sciName")
            obs_dt = record.get("obsDt")

            if lat is None or lng is None or not sci_name or not obs_dt:
                return None

            # Parse date — eBird returns "YYYY-MM-DD HH:MM" or "YYYY-MM-DD"
            try:
                observed_at = datetime.strptime(obs_dt, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                observed_at = datetime.strptime(obs_dt, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            # Quality assessment
            obs_valid = record.get("obsValid", False)
            obs_reviewed = record.get("obsReviewed", False)
            if obs_valid:
                quality_tier = 1  # Validated by reviewer
                quality_grade = "research"
            elif obs_reviewed:
                quality_tier = 2  # Reviewed but not confirmed
                quality_grade = "community"
            else:
                quality_tier = 2  # eBird's automated filters are strong
                quality_grade = "community"

            sub_id = record.get("subId", "")

            return EcologicalObservation(
                id=f"ebird:{sub_id}:{record.get('speciesCode', '')}",
                modality="occurrence",
                location=Location(
                    lat=lat,
                    lng=lng,
                    site_id=record.get("locId"),
                    site_name=record.get("locName"),
                    country=record.get("countryName"),
                    state_province=record.get("subnational1Name"),
                ),
                observed_at=observed_at,
                taxon=TaxonInfo(
                    scientific_name=sci_name,
                    common_name=record.get("comName"),
                    # DECISION: eBird uses its own taxonomy (Clements/eBird).
                    # gbif_id would need a separate lookup via GBIF species match API.
                    # Defer to Phase 2.
                ),
                value={
                    "count": record.get("howMany"),
                    "species_code": record.get("speciesCode"),
                    "checklist_id": sub_id,
                    "location_private": record.get("locationPrivate", False),
                },
                unit="count" if record.get("howMany") else "presence",
                quality=Quality(
                    tier=quality_tier,
                    grade=quality_grade,
                    validated=obs_valid,
                    flags=[],
                ),
                provenance=Provenance(
                    source_api="ebird",
                    source_id=f"{sub_id}:{record.get('speciesCode', '')}",
                    license="eBird Terms of Use",
                    original_url=f"https://ebird.org/checklist/{sub_id}" if sub_id else None,
                    institution_code="CLO",  # Cornell Lab of Ornithology
                ),
                raw=record,
            )
        except Exception:
            return None
