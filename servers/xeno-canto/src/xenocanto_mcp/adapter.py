"""
Xeno-canto Adapter — Bird and Wildlife Sound Recordings.

Queries the Xeno-canto API v2 for audio recordings of bird and
wildlife vocalizations. 1M+ recordings from around the world.

API docs: https://xeno-canto.org/explore/api
No authentication required (v2 removed auth requirement).
Rate limit: soft throttle, be respectful (~1 req/sec).

Quality ratings: A (highest) through E (lowest).
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

XC_API_BASE = "https://xeno-canto.org/api/3/recordings"

# Map Xeno-canto quality ratings (A-E) to our quality tiers (1-4)
QUALITY_MAP = {
    "A": (1, 1.0),    # Excellent — tier 1, full confidence
    "B": (2, 0.85),   # Good
    "C": (2, 0.7),    # Acceptable
    "D": (3, 0.5),    # Poor
    "E": (4, 0.3),    # Very poor
}


class XenoCantoAdapter:
    """Adapter for the Xeno-canto bird sound recording API (v3)."""

    def __init__(self, api_key: str | None = None):
        import os
        self._api_key = api_key or os.environ.get("XC_API_KEY")
        self._client = httpx.AsyncClient(
            timeout=30,
            headers={"Accept": "application/json"},
        )

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_id="xeno-canto",
            name="Xeno-canto",
            description=(
                "The world's largest collection of bird and wildlife sound recordings. "
                "1M+ recordings covering 10,000+ species. Community-contributed with "
                "quality ratings (A-E). Audio files available for download. "
                "Essential for acoustic ecology, soundscape analysis, and species ID."
            ),
            modalities=["acoustic"],
            geographic_coverage="global",
            temporal_coverage_start="2005",
            update_frequency="daily",
            quality_tier=2,
            requires_auth=False,
            license="CC (varies per recording — CC-BY, CC-BY-NC, CC-BY-SA, etc.)",
            homepage_url="https://xeno-canto.org",
            supports_location_search=True,
            supports_taxon_search=True,
            supports_date_range=False,  # XC API doesn't support date range natively
            supports_site_search=False,
        )

    @property
    def id(self) -> str:
        return "xeno-canto"

    async def search(self, params: SearchParams) -> list[EcologicalObservation]:
        """Search Xeno-canto for bird sound recordings."""
        if not self._api_key:
            return []  # v3 requires an API key

        # Build XC v3 query string using search tags
        query_parts = []

        if params.taxon:
            # v3 uses sp: tag for species search
            query_parts.append(f'sp:"{params.taxon}"')

        if params.lat is not None and params.lng is not None:
            # XC supports location search with lat/lng box
            radius_km = params.radius_km or 100
            deg = radius_km / 111.0
            query_parts.append(f"lat:{params.lat - deg}:{params.lat + deg}")
            query_parts.append(f"lon:{params.lng - deg}:{params.lng + deg}")

        if not query_parts:
            return []

        query = " ".join(query_parts)

        try:
            resp = await self._client.get(
                XC_API_BASE,
                params={"query": query, "key": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError:
            return []

        results = []
        recordings = data.get("recordings", [])
        limit = min(params.limit or 20, len(recordings))

        for rec in recordings[:limit]:
            obs = self._to_observation(rec)
            if obs:
                results.append(obs)

        return results

    async def get_by_id(self, xc_id: str) -> Optional[EcologicalObservation]:
        """Fetch a specific recording by Xeno-canto ID."""
        if not self._api_key:
            return None
        try:
            resp = await self._client.get(
                XC_API_BASE,
                params={"query": f"nr:{xc_id}", "key": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            recordings = data.get("recordings", [])
            if recordings:
                return self._to_observation(recordings[0])
        except httpx.HTTPError:
            pass
        return None

    def _to_observation(self, rec: dict) -> Optional[EcologicalObservation]:
        """Convert a Xeno-canto recording to EcologicalObservation."""
        try:
            lat = float(rec.get("lat", 0))
            lng = float(rec.get("lng", 0))
        except (ValueError, TypeError):
            return None

        if lat == 0 and lng == 0:
            return None

        # Parse date
        date_str = rec.get("date", "")
        try:
            observed_at = datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime(1970, 1, 1)
        except ValueError:
            observed_at = datetime(1970, 1, 1)

        xc_id = rec.get("id", "")
        quality_rating = rec.get("q", "C")
        tier, confidence = QUALITY_MAP.get(quality_rating, (3, 0.5))

        # Audio URL — XC provides a download URL
        audio_url = rec.get("file")
        if audio_url and not audio_url.startswith("http"):
            audio_url = f"https:{audio_url}"

        return EcologicalObservation(
            id=f"xeno-canto:{xc_id}",
            modality="acoustic",
            taxon=TaxonInfo(
                scientific_name=rec.get("gen", "") + " " + rec.get("sp", ""),
                common_name=rec.get("en"),
                genus=rec.get("gen"),
            ),
            location=Location(
                lat=lat,
                lng=lng,
                country=rec.get("cnt"),
                site_name=rec.get("loc"),
            ),
            observed_at=observed_at,
            value={
                "recording_type": rec.get("type", ""),
                "duration": rec.get("length", ""),
                "sample_rate": rec.get("smp"),
                "bitrate": rec.get("bitrate"),
                "channels": rec.get("channels"),
                "sonogram_url": rec.get("sono", {}).get("small") if isinstance(rec.get("sono"), dict) else None,
                "recordist": rec.get("rec", ""),
                "remarks": rec.get("rmk", ""),
            },
            unit="audio recording",
            media_url=audio_url,
            media_type="audio/mpeg",
            quality=Quality(
                tier=tier,
                grade="community",
                validated=quality_rating in ("A", "B"),
                confidence=confidence,
                flags=[f"xc_quality:{quality_rating}"],
            ),
            provenance=Provenance(
                source_api="xeno-canto",
                source_id=str(xc_id),
                original_url=f"https://xeno-canto.org/{xc_id}",
                license=rec.get("lic", ""),
                attribution=rec.get("rec", "Unknown recordist"),
                citation_string=(
                    f"{rec.get('rec', 'Unknown')} ({date_str[:4] if date_str else 'n.d.'}). "
                    f"{rec.get('en', rec.get('gen', '') + ' ' + rec.get('sp', ''))}. "
                    f"Xeno-canto recording XC{xc_id}. "
                    f"https://xeno-canto.org/{xc_id}"
                ),
            ),
            raw=rec,
        )
