"""
USGS NWIS Adapter — National Water Information System hydrological data.

Uses the new OGC API (api.waterdata.usgs.gov), not the legacy API.
The legacy API (waterservices.usgs.gov) is being decommissioned early 2027.

Auth: Free API key raises rate limit from ~50 to 1,000 req/hr.
Signup: https://api.waterdata.usgs.gov/signup
Coverage: ~13,500 active real-time stations across US states/territories.
Data: Streamflow, gage height, water temperature, dissolved oxygen, pH, turbidity.
Temporal resolution: 15-minute continuous data, daily statistics.

This is the bridge adapter for Phase 2 watershed monitoring agents.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from math import radians, sin, cos, sqrt, atan2
from typing import Optional

import httpx

from kinship_shared import (
    EcologicalAdapter,
    AdapterCapabilities,
    EcologicalObservation,
    Location,
    Quality,
    Provenance,
    SearchParams,
)
from kinship_shared.retry import http_get_with_retry

logger = logging.getLogger(__name__)

BASE_URL = "https://api.waterdata.usgs.gov/ogcapi/v0"

# Parameter code → human-readable name + unit
PARAM_CODES = {
    "00060": ("Discharge", "ft3/s"),
    "00065": ("Gage height", "ft"),
    "00010": ("Water temperature", "deg C"),
    "00300": ("Dissolved oxygen", "mg/L"),
    "00400": ("pH", "standard units"),
    "63680": ("Turbidity", "FNU"),
    "00095": ("Specific conductance", "uS/cm"),
}

# Default parameters to fetch when no specific ones requested
DEFAULT_PARAMS = ["00060", "00010"]  # discharge + temperature


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _radius_to_bbox(lat: float, lon: float, radius_km: float) -> str:
    """Convert lat/lon + radius to a bounding box string for the USGS API."""
    # Rough approximation: 1 degree lat ≈ 111 km
    delta_lat = radius_km / 111.0
    delta_lon = radius_km / (111.0 * cos(radians(lat)))
    west = lon - delta_lon
    south = lat - delta_lat
    east = lon + delta_lon
    north = lat + delta_lat
    return f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"


class USGSNWISAdapter(EcologicalAdapter):
    """Adapter for the USGS National Water Information System."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("USGS_API_KEY")
        self._site_cache: dict[str, dict] = {}  # monitoring_location_id → site props

    @property
    def id(self) -> str:
        return "usgs-nwis"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_id="usgs-nwis",
            name="USGS NWIS — National Water Information System",
            description=(
                "US Geological Survey water monitoring network with ~13,500 active "
                "real-time stations measuring streamflow, gage height, water temperature, "
                "dissolved oxygen, pH, turbidity, and more. 15-minute continuous data "
                "and daily statistics. Coverage: all US states and territories."
            ),
            modalities=["hydrological"],
            supports_location_search=True,
            supports_taxon_search=False,
            supports_date_range=True,
            supports_site_search=True,
            geographic_coverage="United States and territories",
            temporal_coverage_start="2007-10-01",
            update_frequency="real-time",
            quality_tier=1,
            requires_auth=False,
            rate_limit_per_minute=16,
            license="public-domain",
            homepage_url="https://waterdata.usgs.gov/",
        )

    def _headers(self) -> dict:
        headers = {}
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        return headers

    async def _fetch_site_metadata(self, client: httpx.AsyncClient, site_id: str) -> dict | None:
        """Fetch and cache monitoring location metadata (coordinates, name, etc.)."""
        if site_id in self._site_cache:
            return self._site_cache[site_id]

        resp = await http_get_with_retry(
            client,
            f"{BASE_URL}/collections/monitoring-locations/items",
            params={"monitoring_location_id": site_id, "limit": 1, "f": "json"},
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        features = data.get("features", [])
        if not features:
            return None

        feature = features[0]
        self._site_cache[site_id] = feature
        return feature

    async def _find_sites_near(
        self, client: httpx.AsyncClient, lat: float, lon: float, radius_km: float
    ) -> list[dict]:
        """Find monitoring locations within radius_km of a point."""
        bbox = _radius_to_bbox(lat, lon, radius_km)
        resp = await http_get_with_retry(
            client,
            f"{BASE_URL}/collections/monitoring-locations/items",
            params={"bbox": bbox, "limit": 100, "f": "json"},
        )
        if resp.status_code != 200:
            return []
        data = resp.json()

        # Post-filter by haversine distance
        sites = []
        for feature in data.get("features", []):
            geom = feature.get("geometry")
            if not geom or not geom.get("coordinates"):
                continue
            site_lon, site_lat = geom["coordinates"][:2]
            dist = _haversine_km(lat, lon, site_lat, site_lon)
            if dist <= radius_km:
                feature["_distance_km"] = dist
                self._site_cache[feature["properties"]["id"]] = feature
                sites.append(feature)

        sites.sort(key=lambda f: f.get("_distance_km", 999))
        return sites

    async def search(self, params: SearchParams) -> list[EcologicalObservation]:
        """
        Search USGS NWIS for water monitoring data.

        Strategy:
        1. Find monitoring locations near the search point (or by site_id)
        2. Fetch recent continuous data from those sites
        3. Convert to EcologicalObservation
        """
        headers = self._headers()
        observations = []

        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            # Find sites
            sites = []
            if params.site_id:
                site_id = params.site_id
                if not site_id.startswith("USGS-"):
                    site_id = f"USGS-{site_id}"
                site = await self._fetch_site_metadata(client, site_id)
                if site:
                    sites = [site]
            elif params.lat is not None and params.lng is not None:
                radius = min(params.radius_km or 50, 200)
                sites = await self._find_sites_near(client, params.lat, params.lng, radius)

            if not sites:
                return []

            # Fetch data from up to 5 nearest sites
            for site_feature in sites[:5]:
                site_props = site_feature["properties"]
                site_id = site_props["id"]

                query_params: dict = {
                    "monitoring_location_id": site_id,
                    "limit": min(params.limit, 100),
                    "f": "json",
                }

                # Add parameter codes
                query_params["parameter_code"] = ",".join(DEFAULT_PARAMS)

                # Add date range
                if params.start_date and params.end_date:
                    query_params["datetime"] = f"{params.start_date}/{params.end_date}"

                # Use daily values for date ranges > 7 days, continuous for recent
                collection = "daily" if params.start_date and params.end_date else "continuous"
                if collection == "daily":
                    query_params["statistic_id"] = "00003"  # mean

                resp = await http_get_with_retry(
                    client,
                    f"{BASE_URL}/collections/{collection}/items",
                    params=query_params,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()

                for feature in data.get("features", []):
                    obs = self._feature_to_observation(feature, site_feature, collection)
                    if obs:
                        observations.append(obs)

        return observations[:params.limit]

    async def get_by_id(self, source_id: str) -> Optional[EcologicalObservation]:
        """USGS doesn't support fetching individual observations by our composite ID."""
        return None

    def _feature_to_observation(
        self, feature: dict, site_feature: dict, collection: str
    ) -> Optional[EcologicalObservation]:
        """Convert a USGS OGC API feature to EcologicalObservation."""
        try:
            props = feature.get("properties", {})
            site_props = site_feature.get("properties", {})
            geom = site_feature.get("geometry", {})

            # Skip sentinel values
            value_str = props.get("value")
            if not value_str or value_str in ("-999999", "-999999.00"):
                return None

            try:
                measurement = float(value_str)
            except (ValueError, TypeError):
                return None

            # Parse time
            time_str = props.get("time")
            if not time_str:
                return None
            observed_at = datetime.fromisoformat(time_str)
            if observed_at.tzinfo is None:
                observed_at = observed_at.replace(tzinfo=timezone.utc)

            # Location from site metadata
            coords = geom.get("coordinates", [None, None])
            if coords[0] is None:
                return None

            param_code = props.get("parameter_code", "")
            param_name, param_unit = PARAM_CODES.get(param_code, (f"param_{param_code}", "unknown"))

            # Convert altitude from feet to meters
            altitude_ft = site_props.get("altitude")
            elevation_m = altitude_ft * 0.3048 if altitude_ft else None

            site_num = site_props.get("monitoring_location_number", "")
            approval = props.get("approval_status", "")

            return EcologicalObservation(
                id=f"usgs-nwis:{site_props.get('id', '')}:{param_code}:{time_str}",
                modality="hydrological",
                location=Location(
                    lat=coords[1],
                    lng=coords[0],
                    elevation_m=elevation_m,
                    site_id=site_num,
                    site_name=site_props.get("monitoring_location_name"),
                    country_code="US",
                    state_province=site_props.get("state_name"),
                    watershed_id=site_props.get("hydrologic_unit_code"),
                ),
                observed_at=observed_at,
                temporal_resolution="15min" if collection == "continuous" else "daily",
                value={
                    "parameter_code": param_code,
                    "parameter_name": param_name,
                    "measurement": measurement,
                    "statistic": props.get("statistic_id"),
                    "drainage_area_sq_mi": site_props.get("drainage_area"),
                },
                unit=param_unit,
                quality=Quality(
                    tier=1,
                    grade="research",
                    validated=(approval == "Approved"),
                    flags=["provisional"] if approval != "Approved" else [],
                ),
                provenance=Provenance(
                    source_api="usgs-nwis",
                    source_id=props.get("id", ""),
                    original_url=f"https://waterdata.usgs.gov/monitoring-location/{site_num}",
                    license="public-domain",
                    attribution="U.S. Geological Survey, National Water Information System",
                    institution_code="USGS",
                    sensor_id=props.get("time_series_id"),
                    collection_method="automated_sensor",
                ),
                raw=props,
            )
        except Exception:
            return None
