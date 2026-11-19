"""
NEON (National Ecological Observatory Network) adapter for Kinship Earth.

NEON operates 81 field sites across 20 ecoclimatic domains in the US,
providing 180+ standardized ecological data products. All data is
research-grade (quality tier 1) and freely accessible.

API base: https://data.neonscience.org/api/v0
Auth: No auth required for basic access (optional token raises rate limits)
Rate limit: 200 req/day (no token), 1000 req/day (with token)

NEON citation requirement: always include the dataset DOI and citation string.
"""

from __future__ import annotations

import csv
import io
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
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
    http_get_with_retry,
)

logger = logging.getLogger(__name__)

NEON_API_BASE = "https://data.neonscience.org/api/v0"
NEON_PORTAL_BASE = "https://data.neonscience.org"


class NeonAdapter(EcologicalAdapter):
    """Adapter for the NEON REST API."""

    def __init__(self, api_token: Optional[str] = None):
        self._token = api_token
        self._headers = {"X-API-Token": api_token} if api_token else {}

    @property
    def id(self) -> str:
        return "neonscience"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_id="neonscience",
            name="NEON — National Ecological Observatory Network",
            description=(
                "Continental-scale ecological observatory with 81 field sites across "
                "20 US ecoclimatic domains. Provides 180+ standardized data products "
                "including soil sensors, bird surveys, aquatic monitoring, eddy covariance "
                "flux towers, airborne hyperspectral imagery, and bioacoustics. "
                "All data is research-grade and freely accessible."
            ),
            modalities=["sensor", "occurrence", "acoustic", "geospatial"],
            supports_location_search=True,
            supports_taxon_search=False,
            supports_date_range=True,
            supports_site_search=True,
            geographic_coverage="United States (81 sites, 20 ecoclimatic domains)",
            temporal_coverage_start="2012-01-01",
            update_frequency="varies by product (real-time to annual)",
            quality_tier=1,
            requires_auth=False,
            rate_limit_per_minute=None,  # daily limit: 200/day (no token)
            license="CC-BY-4.0",
            homepage_url="https://www.neonscience.org",
        )

    async def list_sites(self) -> list[dict]:
        """Fetch all 81 NEON field sites."""
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            resp = await http_get_with_retry(client, f"{NEON_API_BASE}/sites")
            logger.info("NEON list_sites HTTP response: status=%d", resp.status_code)
            resp.raise_for_status()
            return resp.json().get("data", [])

    async def get_site(self, site_code: str) -> Optional[dict]:
        """Fetch details for a single NEON site by its code (e.g. 'WREF')."""
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            resp = await http_get_with_retry(client, f"{NEON_API_BASE}/sites/{site_code.upper()}")
            # NEON returns 404 for unknown but valid-format codes,
            # and 400 for malformed codes (e.g. codes that are too long).
            # Both mean "not found" from our perspective.
            if resp.status_code in (400, 404):
                return None
            resp.raise_for_status()
            return resp.json().get("data")

    async def list_data_products(self) -> list[dict]:
        """Fetch all available NEON data products."""
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            resp = await http_get_with_retry(client, f"{NEON_API_BASE}/products")
            resp.raise_for_status()
            return resp.json().get("data", [])

    async def get_data_product(self, product_code: str) -> Optional[dict]:
        """Fetch metadata for a single NEON data product."""
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            resp = await http_get_with_retry(client, f"{NEON_API_BASE}/products/{product_code}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json().get("data")

    async def get_product_data(
        self,
        product_code: str,
        site_code: str,
        year_month: str,
        table: str = "countdata",
        cache_dir: Optional[Path] = None,
    ) -> list[dict]:
        """Download and parse the basic-package CSV tables for one product/site/month.

        # Verified against NEON Data API documentation:
        # - GET /v0/data/{product}/{site}/{year_month} returns all files in one response (no pagination).
        # - basic-package files are named NEON.D*.{site}.{product}.{table}.{period}.basic.*.csv
        # - Maximum response: all files for a site-month (typically 5–20 files).
        # - DP1.10003.001 daily record begins 2013 at most sites; temporal_coverage_start in
        #   capabilities reflects 2012 (observatory open date), but tabular landbird data starts 2013.

        Args:
            product_code: NEON product code, e.g. "DP1.10003.001".
            site_code: Four-letter NEON site code, e.g. "WREF".
            year_month: "YYYY-MM", e.g. "2023-06".
            table: Substring to match in CSV filenames (default "countdata").
            cache_dir: Directory to cache downloaded CSV files. If None, no caching.
        """
        url = f"{NEON_API_BASE}/data/{product_code}/{site_code.upper()}/{year_month}"
        async with httpx.AsyncClient(headers=self._headers, timeout=60) as client:
            resp = await http_get_with_retry(client, url)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            file_list = resp.json().get("data", {}).get("files", [])

        rows: list[dict] = []
        for file_info in file_list:
            name = file_info.get("name", "")
            # Select basic-package CSV files matching the table name.
            # Skip expanded files, variables tables, readme, sensor positions, etc.
            if not name.endswith(".csv"):
                continue
            if table not in name:
                continue
            if "expanded" in name:
                continue

            cached_path = (cache_dir / name) if cache_dir else None
            if cached_path and cached_path.exists():
                csv_text = cached_path.read_text(encoding="utf-8")
            else:
                file_url = file_info.get("url", "")
                if not file_url:
                    continue
                time.sleep(0.5)  # respect NEON sequential-request guidance
                async with httpx.AsyncClient(headers=self._headers, timeout=120) as dl:
                    r = await http_get_with_retry(dl, file_url)
                    r.raise_for_status()
                    csv_text = r.text
                if cached_path:
                    cached_path.parent.mkdir(parents=True, exist_ok=True)
                    cached_path.write_text(csv_text, encoding="utf-8")

            reader = csv.DictReader(io.StringIO(csv_text))
            rows.extend(reader)

        return rows

    async def search(self, params: SearchParams) -> list[EcologicalObservation]:
        """
        Search NEON sites as observations.
        For now, returns site-level observations filtered by location proximity.
        """
        logger.info("NEON search: lat=%s, lng=%s, radius_km=%s", params.lat, params.lng, params.radius_km)
        sites = await self.list_sites()
        results = []

        for site in sites:
            # Apply location filter if provided
            if params.lat is not None and params.lng is not None and params.radius_km is not None:
                site_lat = site.get("siteLatitude")
                site_lng = site.get("siteLongitude")
                if site_lat is None or site_lng is None:
                    continue
                if not _within_radius(params.lat, params.lng, site_lat, site_lng, params.radius_km):
                    continue

            # Apply site_id filter if provided
            if params.site_id and site.get("siteCode", "").upper() != params.site_id.upper():
                continue

            obs = _site_to_observation(site)
            if obs:
                results.append(obs)

            if len(results) >= params.limit:
                break

        if not results:
            logger.warning("NEON search returned no sites within radius_km=%s of lat=%s, lng=%s", params.radius_km, params.lat, params.lng)
        return results

    async def get_by_id(self, source_id: str) -> Optional[EcologicalObservation]:
        """
        Fetch a NEON site by its site code.
        source_id should be the site code (e.g. 'WREF').
        """
        site = await self.get_site(source_id)
        if not site:
            return None
        return _site_to_observation(site)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _site_to_observation(site: dict) -> Optional[EcologicalObservation]:
    """Convert a NEON site record to an EcologicalObservation."""
    site_code = site.get("siteCode")
    lat = site.get("siteLatitude")
    lng = site.get("siteLongitude")

    if not site_code or lat is None or lng is None:
        return None

    return EcologicalObservation(
        id=f"neonscience:{site_code}",
        modality="sensor",
        location=Location(
            lat=lat,
            lng=lng,
            elevation_m=site.get("siteElevation"),
            site_id=site_code,
            site_name=site.get("siteName"),
            state_province=site.get("stateCode"),
            country="United States",
            country_code="US",
        ),
        observed_at=datetime(2012, 1, 1, tzinfo=timezone.utc),  # NEON operational start
        value={
            "site_code": site_code,
            "site_name": site.get("siteName"),
            "site_type": site.get("siteType"),
            "domain_code": site.get("domainCode"),
            "domain_name": site.get("domainName"),
            "state_code": site.get("stateCode"),
            "data_products_available": len(site.get("dataProducts", [])),
        },
        quality=Quality(
            tier=1,
            grade="research",
            validated=True,
        ),
        provenance=Provenance(
            source_api="neonscience",
            source_id=site_code,
            original_url=f"https://www.neonscience.org/field-sites/{site_code.lower()}",
            license="CC-BY-4.0",
            attribution="National Ecological Observatory Network (NEON)",
            citation_string=(
                f"National Ecological Observatory Network. {site.get('siteName', site_code)}. "
                f"NEON Field Site. https://www.neonscience.org/field-sites/{site_code.lower()}"
            ),
            institution_code="NEON",
        ),
        raw=site,
    )


def _within_radius(lat1: float, lng1: float, lat2: float, lng2: float, radius_km: float) -> bool:
    """Simple Haversine distance check."""
    import math
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a)) <= radius_km
