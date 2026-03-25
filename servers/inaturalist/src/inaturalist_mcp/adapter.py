"""
iNaturalist Adapter — community biodiversity observation data.

API: https://api.inaturalist.org/v1/docs/
Auth: None required for read-only access
Rate limit: 60 requests/minute (100 req/min with API key)
Coverage: Global, 200M+ observations, all taxa (plants, animals, fungi)

iNaturalist includes observations from the SEEK app (camera-based species
identification using computer vision). SEEK observations can optionally be
submitted to iNaturalist, where they undergo community identification.
"""

from __future__ import annotations

from datetime import datetime, timezone
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

BASE_URL = "https://api.inaturalist.org/v1"


class INaturalistAdapter(EcologicalAdapter):
    """Adapter for the iNaturalist API."""

    @property
    def id(self) -> str:
        return "inaturalist"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_id="inaturalist",
            name="iNaturalist",
            description=(
                "A global community biodiversity platform with over 200 million "
                "observations of plants, animals, fungi, and other organisms. "
                "Observations are photo-verified through community consensus. "
                "Includes data from the SEEK app (AI-powered species identification "
                "via camera). Research-grade observations are shared with GBIF."
            ),
            modalities=["occurrence", "visual"],
            supports_location_search=True,
            supports_taxon_search=True,
            supports_date_range=True,
            supports_site_search=False,
            geographic_coverage="Global",
            temporal_coverage_start="2008-01-01",
            update_frequency="real-time",
            quality_tier=2,  # Mix of research-grade and casual
            requires_auth=False,
            rate_limit_per_minute=60,
            license="Varies per observation (CC0, CC-BY, CC-BY-NC, etc.)",
            homepage_url="https://www.inaturalist.org",
        )

    async def search(self, params: SearchParams) -> list[EcologicalObservation]:
        """
        Search iNaturalist for biodiversity observations.

        iNaturalist API supports rich filtering:
        - Location (lat/lng/radius or bounding box)
        - Taxon (by name or taxon_id)
        - Date range (d1/d2 parameters)
        - Quality grade (research, needs_id, casual)
        - Photos, sounds, identification status
        """
        api_params: dict = {
            "per_page": min(params.limit, 200),
            "order": "desc",
            "order_by": "observed_on",
        }

        # Location filter
        if params.lat is not None and params.lng is not None:
            api_params["lat"] = params.lat
            api_params["lng"] = params.lng
            api_params["radius"] = min(params.radius_km or 25, 500)  # km

        # Taxon filter
        if params.taxon:
            api_params["taxon_name"] = params.taxon

        # Iconic taxon group filter (Aves, Mammalia, Insecta, Plantae, etc.)
        if getattr(params, 'iconic_taxon', None):
            api_params["iconic_taxon_name"] = params.iconic_taxon

        # Date range filter
        if params.start_date:
            api_params["d1"] = params.start_date
        if params.end_date:
            api_params["d2"] = params.end_date

        # Quality filter — default to verifiable observations
        if params.quality_tier_min and params.quality_tier_min <= 1:
            api_params["quality_grade"] = "research"
        else:
            api_params["verifiable"] = "true"

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await http_get_with_retry(client, f"{BASE_URL}/observations", params=api_params)
            resp.raise_for_status()
            data = resp.json()

        observations = []
        if isinstance(data, dict) and "results" in data:
            for record in data["results"]:
                obs = self._record_to_observation(record)
                if obs is not None:
                    observations.append(obs)

        return observations[:params.limit]

    async def get_by_id(self, source_id: str) -> Optional[EcologicalObservation]:
        """Fetch a specific iNaturalist observation by ID."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await http_get_with_retry(client, f"{BASE_URL}/observations/{source_id}")
            resp.raise_for_status()
            data = resp.json()
        if isinstance(data, dict) and "results" in data and data["results"]:
            return self._record_to_observation(data["results"][0])
        return None

    def _record_to_observation(self, record: dict) -> Optional[EcologicalObservation]:
        """
        Convert an iNaturalist API observation to EcologicalObservation.

        iNaturalist response fields (key ones):
        - id: observation ID
        - taxon: nested object with name, preferred_common_name, rank,
          ancestors (full taxonomy), iconic_taxon_name
        - geojson: {type: "Point", coordinates: [lng, lat]}
        - location: "lat,lng" string
        - observed_on: date string
        - time_observed_at: ISO datetime
        - quality_grade: "research", "needs_id", "casual"
        - photos: list of photo objects with url field
        - sounds: list of sound objects
        - uri: canonical URL
        - user: observer info
        - identifications_count: number of community IDs
        - oauth_application_id: which app created this (SEEK = specific ID)
        """
        try:
            obs_id = record.get("id")
            if obs_id is None:
                return None

            # Location
            geojson = record.get("geojson")
            if geojson and geojson.get("coordinates"):
                lng, lat = geojson["coordinates"]
            elif record.get("location"):
                parts = record["location"].split(",")
                lat, lng = float(parts[0]), float(parts[1])
            else:
                return None  # No location, skip

            # Date
            time_str = record.get("time_observed_at") or record.get("observed_on")
            if not time_str:
                return None
            try:
                observed_at = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            except ValueError:
                observed_at = datetime.strptime(time_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)

            # Taxon
            taxon_data = record.get("taxon")
            taxon_info = None
            if taxon_data:
                # Build taxonomy from ancestors
                ancestors = {a.get("rank"): a.get("name") for a in (taxon_data.get("ancestors") or [])}
                taxon_info = TaxonInfo(
                    scientific_name=taxon_data.get("name", "Unknown"),
                    common_name=taxon_data.get("preferred_common_name"),
                    gbif_id=taxon_data.get("gbif_id"),  # DECISION: iNat sometimes includes this
                    rank=taxon_data.get("rank"),
                    kingdom=ancestors.get("kingdom"),
                    phylum=ancestors.get("phylum"),
                    class_=ancestors.get("class"),
                    order=ancestors.get("order"),
                    family=ancestors.get("family"),
                )

            # Quality mapping
            quality_grade_str = record.get("quality_grade", "casual")
            quality_map = {
                "research": (1, "research"),
                "needs_id": (2, "community"),
                "casual": (3, "casual"),
            }
            tier, grade = quality_map.get(quality_grade_str, (3, "casual"))

            # Photos
            photos = record.get("photos") or []
            media_url = None
            media_type = None
            if photos:
                # iNat photo URLs: replace "square" with "medium" for better resolution
                photo_url = photos[0].get("url", "")
                media_url = photo_url.replace("/square.", "/medium.")
                media_type = "image/jpeg"

            # License
            license_code = record.get("license_code") or "all-rights-reserved"

            # DECISION: Detect SEEK-originated observations
            # SEEK's oauth_application_id is typically a specific value.
            # For now, note the app ID for future filtering.
            oauth_app_id = record.get("oauth_application_id")

            return EcologicalObservation(
                id=f"inaturalist:{obs_id}",
                modality="occurrence",
                location=Location(
                    lat=lat,
                    lng=lng,
                    country=record.get("place_guess"),
                ),
                observed_at=observed_at,
                taxon=taxon_info,
                value={
                    "quality_grade": quality_grade_str,
                    "identifications_count": record.get("identifications_count", 0),
                    "photos_count": len(photos),
                    "sounds_count": len(record.get("sounds") or []),
                    "oauth_application_id": oauth_app_id,
                    "iconic_taxon": taxon_data.get("iconic_taxon_name") if taxon_data else None,
                },
                unit="presence",
                media_url=media_url,
                media_type=media_type,
                quality=Quality(
                    tier=tier,
                    grade=grade,
                    validated=(quality_grade_str == "research"),
                    flags=[],
                ),
                provenance=Provenance(
                    source_api="inaturalist",
                    source_id=str(obs_id),
                    license=license_code,
                    original_url=record.get("uri") or f"https://www.inaturalist.org/observations/{obs_id}",
                ),
                raw=record,
            )
        except Exception:
            return None
