"""
OBIS (Ocean Biodiversity Information System) adapter for Kinship Earth.

OBIS aggregates 168M+ marine species occurrence records from 166K+ species
across 5,000+ datasets worldwide. It is the most comprehensive open repository
of marine biodiversity data. No authentication required.

API base: https://api.obis.org/v3
Auth: None required
Darwin Core native: decimalLatitude/Longitude, scientificName, eventDate, full taxonomy
Pagination: cursor-based via `after=<last_record_uuid>` (NOT offset-based)

OBIS citation: data.obis.org — individual records carry per-record license fields.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

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
    http_get_with_retry,
)

OBIS_API_BASE = "https://api.obis.org/v3"


class OBISAdapter(EcologicalAdapter):
    """Adapter for the OBIS REST API v3."""

    @property
    def id(self) -> str:
        return "obis"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_id="obis",
            name="OBIS — Ocean Biodiversity Information System",
            description=(
                "The world's largest open-access repository of marine species occurrence data. "
                "168M+ occurrence records spanning 166K+ marine species, aggregated from "
                "5,000+ datasets contributed by institutions worldwide. Covers the entire "
                "marine biodiversity domain: fish, mammals, invertebrates, algae, plankton. "
                "Darwin Core native. No authentication required."
            ),
            modalities=["occurrence"],
            supports_location_search=True,
            supports_taxon_search=True,
            supports_date_range=True,
            supports_site_search=False,
            geographic_coverage="Global oceans",
            temporal_coverage_start="1800-01-01",
            update_frequency="continuous",
            quality_tier=2,
            requires_auth=False,
            rate_limit_per_minute=None,
            license="varies per record (CC0-1.0, CC-BY, CC-BY-NC)",
            homepage_url="https://obis.org",
        )

    async def search(self, params: SearchParams) -> list[EcologicalObservation]:
        """Search OBIS occurrence records."""
        logger.info("OBIS search: taxon=%s, lat=%s, lng=%s, radius_km=%s", params.taxon, params.lat, params.lng, params.radius_km)
        query: dict = {"size": params.limit}

        has_geo = params.lat is not None and params.lng is not None and params.radius_km is not None

        if has_geo:
            if params.taxon:
                # OBIS ignores lat/lon/radius when scientificname is set.
                # Use a WKT bounding box via the `geometry` param instead,
                # then apply exact radius filtering client-side.
                bbox = _radius_to_bbox(params.lat, params.lng, params.radius_km)
                query["geometry"] = bbox
                logger.info("OBIS: using geometry bbox (taxon+geo combo): %s", bbox)
            else:
                query["lat"] = params.lat
                query["lon"] = params.lng
                query["radius"] = params.radius_km

        if params.taxon:
            query["scientificname"] = params.taxon

        if params.taxon_id:
            query["taxonid"] = params.taxon_id

        if params.start_date:
            query["startdate"] = params.start_date

        if params.end_date:
            query["enddate"] = params.end_date

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await http_get_with_retry(client, f"{OBIS_API_BASE}/occurrence", params=query)
            logger.info("OBIS HTTP response: status=%d", resp.status_code)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for record in data.get("results", []):
            obs = _occurrence_to_observation(record)
            if obs:
                # Apply client-side radius filter when location params are given.
                # OBIS server-side geo filtering is unreliable when combined with
                # taxon filters — we enforce accuracy ourselves.
                if (params.lat is not None and params.lng is not None
                        and params.radius_km is not None):
                    if not _within_radius(
                        params.lat, params.lng,
                        obs.location.lat, obs.location.lng,
                        params.radius_km,
                    ):
                        continue
                results.append(obs)

        if not results:
            logger.warning("OBIS search returned empty results for taxon=%s", params.taxon)
        return results

    async def get_by_id(self, source_id: str) -> Optional[EcologicalObservation]:
        """Fetch a single OBIS occurrence by its UUID."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await http_get_with_retry(client, f"{OBIS_API_BASE}/occurrence/{source_id}")
            if resp.status_code in (400, 404):
                return None
            resp.raise_for_status()
            data = resp.json()

        # OBIS returns the occurrence directly (not nested under 'results')
        if not data:
            return None
        return _occurrence_to_observation(data)

    async def get_statistics(self) -> dict:
        """Fetch OBIS-wide statistics: total records, species, datasets."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await http_get_with_retry(client, f"{OBIS_API_BASE}/statistics")
            resp.raise_for_status()
            return resp.json()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _radius_to_bbox(lat: float, lng: float, radius_km: float) -> str:
    """Convert a center point + radius to a WKT POLYGON bounding box for OBIS geometry param."""
    # Approximate degrees per km at this latitude
    lat_delta = radius_km / 111.0
    lng_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    min_lat = max(lat - lat_delta, -90)
    max_lat = min(lat + lat_delta, 90)
    min_lng = max(lng - lng_delta, -180)
    max_lng = min(lng + lng_delta, 180)
    return f"POLYGON(({min_lng} {min_lat},{max_lng} {min_lat},{max_lng} {max_lat},{min_lng} {max_lat},{min_lng} {min_lat}))"


def _within_radius(lat1: float, lng1: float, lat2: float, lng2: float, radius_km: float) -> bool:
    """Haversine distance check — returns True if (lat2, lng2) is within radius_km of (lat1, lng1)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a)) <= radius_km


def _parse_observed_at(record: dict) -> datetime:
    """Parse eventDate from the record, with fallback to date_mid."""
    event_date = record.get("eventDate") or record.get("date_start")
    if event_date:
        try:
            # OBIS eventDate is ISO 8601; may be date-only or full datetime
            if isinstance(event_date, str):
                # Normalise: add time if missing
                if "T" not in event_date and len(event_date) == 10:
                    event_date = event_date + "T00:00:00"
                return datetime.fromisoformat(event_date.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

    # Fallback: date_mid is Unix milliseconds
    date_mid = record.get("date_mid")
    if date_mid is not None:
        try:
            return datetime.fromtimestamp(int(date_mid) / 1000, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            pass

    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _occurrence_to_observation(record: dict) -> Optional[EcologicalObservation]:
    """Convert an OBIS occurrence record to an EcologicalObservation."""
    occurrence_id = record.get("id") or record.get("occurrenceID")
    lat = record.get("decimalLatitude")
    lng = record.get("decimalLongitude")
    scientific_name = record.get("scientificName") or record.get("species")

    # All three are required
    if not occurrence_id or lat is None or lng is None or not scientific_name:
        return None

    # --- Taxon ---
    taxon = TaxonInfo(
        scientific_name=scientific_name,
        common_name=record.get("vernacularName"),
        rank=record.get("taxonRank"),
        kingdom=record.get("kingdom"),
        phylum=record.get("phylum"),
        **{"class": record.get("class")},
        order=record.get("order"),
        family=record.get("family"),
    )

    # --- Location ---
    location = Location(
        lat=lat,
        lng=lng,
        uncertainty_m=record.get("coordinateUncertaintyInMeters"),
        country=record.get("country"),
        state_province=record.get("stateProvince"),
    )

    # --- Observed at ---
    observed_at = _parse_observed_at(record)

    # --- Quality flags ---
    flags = record.get("flags") or []
    if isinstance(flags, str):
        flags = [flags]

    # OBIS identificationVerificationStatus → validated
    verification = record.get("identificationVerificationStatus", "")
    validated = isinstance(verification, str) and verification.lower() == "verified"

    quality = Quality(
        tier=2,
        grade="community",
        validated=validated,
        flags=flags if flags else None,
    )

    # --- Provenance ---
    record_license = record.get("license") or "unknown"
    dataset_id = record.get("datasetID") or record.get("dataset_id")
    institution_code = record.get("institutionCode")

    provenance = Provenance(
        source_api="obis",
        source_id=str(occurrence_id),
        license=record_license,
        dataset_id=dataset_id,
        institution_code=institution_code,
        attribution=(
            f"OBIS ({record.get('institutionCode', 'contributor')}). "
            f"Retrieved from Ocean Biodiversity Information System, obis.org."
        ),
        original_url=f"https://obis.org/occurrence/{occurrence_id}",
    )

    # --- Value payload ---
    value = {
        "basis_of_record": record.get("basisOfRecord"),
        "occurrence_status": record.get("occurrenceStatus"),
        "depth_m": record.get("depth"),
        "minimum_depth_m": record.get("minimumDepthInMeters"),
        "maximum_depth_m": record.get("maximumDepthInMeters"),
        "sst": record.get("sst"),
        "flags": flags,
        "individual_count": record.get("individualCount"),
    }

    return EcologicalObservation(
        id=f"obis:{occurrence_id}",
        modality="occurrence",
        taxon=taxon,
        location=location,
        observed_at=observed_at,
        value=value,
        quality=quality,
        provenance=provenance,
        raw=record,
    )
