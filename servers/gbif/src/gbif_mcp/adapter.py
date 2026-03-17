"""
GBIF Adapter — Global Biodiversity Information Facility.

Queries the GBIF Occurrence API for species occurrence records.
GBIF aggregates 2.8B+ records from 90,000+ datasets worldwide,
including eBird, iNaturalist, herbarium collections, museum specimens,
and national biodiversity databases.

API docs: https://www.gbif.org/developer/occurrence
No authentication required for read access.
Rate limit: ~1 req/sec (soft, be respectful).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx

from kinship_shared import (
    EcologicalObservation,
    Location,
    Provenance,
    Quality,
    SearchParams,
    TaxonInfo,
    AdapterCapabilities,
)

GBIF_API_BASE = "https://api.gbif.org/v1"


class GBIFAdapter:
    """Adapter for the GBIF Occurrence Search API."""

    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=GBIF_API_BASE,
            timeout=30,
            headers={"Accept": "application/json"},
        )

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_id="gbif",
            name="GBIF (Global Biodiversity Information Facility)",
            description=(
                "The world's largest biodiversity data aggregator. 2.8B+ occurrence records "
                "from 90,000+ datasets including museum specimens, herbarium records, "
                "citizen science (eBird, iNaturalist), and national monitoring programs. "
                "Covers all taxonomic groups globally."
            ),
            modalities=["occurrence"],
            geographic_coverage="global",
            temporal_coverage_start="1500",
            update_frequency="daily",
            quality_tier=2,
            requires_auth=False,
            license="CC-BY-4.0 / CC0 (varies by dataset)",
            homepage_url="https://www.gbif.org",
            supports_location_search=True,
            supports_taxon_search=True,
            supports_date_range=True,
            supports_site_search=False,
        )

    @property
    def id(self) -> str:
        return "gbif"

    async def search(self, params: SearchParams) -> list[EcologicalObservation]:
        """Search GBIF occurrence records."""
        query: dict = {
            "limit": min(params.limit or 20, 300),
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
        }

        if params.taxon:
            # First resolve the species name to a GBIF taxon key
            taxon_key = await self._resolve_taxon(params.taxon)
            if taxon_key:
                query["taxonKey"] = taxon_key
            else:
                query["scientificName"] = params.taxon

        if params.lat is not None and params.lng is not None:
            radius_km = params.radius_km or 200
            # GBIF uses a bounding box via decimalLatitude/decimalLongitude ranges
            deg_offset = radius_km / 111.0  # rough km-to-degrees
            query["decimalLatitude"] = f"{params.lat - deg_offset},{params.lat + deg_offset}"
            query["decimalLongitude"] = f"{params.lng - deg_offset},{params.lng + deg_offset}"

        if params.start_date:
            query["eventDate"] = params.start_date
            if params.end_date:
                query["eventDate"] = f"{params.start_date},{params.end_date}"

        try:
            resp = await self._client.get("/occurrence/search", params=query)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError:
            return []

        results = []
        for record in data.get("results", []):
            obs = self._to_observation(record)
            if obs:
                results.append(obs)

        return results

    async def get_by_id(self, gbif_key: str) -> Optional[EcologicalObservation]:
        """Fetch a specific GBIF occurrence by key."""
        try:
            resp = await self._client.get(f"/occurrence/{gbif_key}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return self._to_observation(resp.json())
        except httpx.HTTPError:
            return None

    async def _resolve_taxon(self, name: str) -> Optional[int]:
        """Resolve a species name to a GBIF taxon key using the Species API."""
        try:
            resp = await self._client.get("/species/match", params={"name": name})
            resp.raise_for_status()
            data = resp.json()
            if data.get("matchType") != "NONE" and data.get("usageKey"):
                return data["usageKey"]
        except httpx.HTTPError:
            pass
        return None

    def _to_observation(self, record: dict) -> Optional[EcologicalObservation]:
        """Convert a GBIF occurrence record to EcologicalObservation."""
        lat = record.get("decimalLatitude")
        lng = record.get("decimalLongitude")
        if lat is None or lng is None:
            return None

        # Parse event date
        event_date_str = record.get("eventDate", "")
        try:
            if len(event_date_str) >= 10:
                observed_at = datetime.fromisoformat(event_date_str[:10])
            else:
                observed_at = datetime(1970, 1, 1)
        except (ValueError, TypeError):
            observed_at = datetime(1970, 1, 1)

        gbif_key = str(record.get("key", ""))

        # Build taxonomy
        taxon = TaxonInfo(
            scientific_name=record.get("scientificName", record.get("species", "")),
            common_name=record.get("vernacularName"),
            gbif_id=record.get("taxonKey"),
            rank=record.get("taxonRank", "").lower() or None,
            kingdom=record.get("kingdom"),
            phylum=record.get("phylum"),
            class_name=record.get("class"),
            order=record.get("order"),
            family=record.get("family"),
            genus=record.get("genus"),
        )

        # Quality — GBIF has basisOfRecord and issue flags
        basis = record.get("basisOfRecord", "UNKNOWN")
        issues = record.get("issues", [])
        grade = "research" if basis in ("PRESERVED_SPECIMEN", "HUMAN_OBSERVATION", "MACHINE_OBSERVATION") else "community"
        tier = 2 if not issues else 3

        return EcologicalObservation(
            id=f"gbif:{gbif_key}",
            modality="occurrence",
            taxon=taxon,
            location=Location(
                lat=lat,
                lng=lng,
                elevation_m=record.get("elevation"),
                country_code=record.get("countryCode"),
                country=record.get("country"),
                state_province=record.get("stateProvince"),
            ),
            observed_at=observed_at,
            value={
                "count": record.get("individualCount"),
                "basis_of_record": basis,
                "institution": record.get("institutionCode"),
                "collection": record.get("collectionCode"),
                "catalog_number": record.get("catalogNumber"),
            },
            quality=Quality(
                tier=tier,
                grade=grade,
                validated=basis == "PRESERVED_SPECIMEN",
                confidence=0.9 if not issues else 0.7,
                flags=issues[:5] if issues else [],
            ),
            provenance=Provenance(
                source_api="gbif",
                source_id=gbif_key,
                original_url=f"https://www.gbif.org/occurrence/{gbif_key}",
                license=record.get("license", ""),
                attribution=record.get("publishingOrgKey", ""),
                dataset_id=record.get("datasetKey"),
                institution_code=record.get("institutionCode"),
                citation_string=(
                    f"GBIF.org ({datetime.now().year}). GBIF Occurrence Download. "
                    f"https://www.gbif.org/occurrence/{gbif_key}"
                ),
            ),
            raw=record,
        )
