"""
SoilGrids Adapter — ISRIC global soil property data at 250m resolution.

API: https://rest.isric.org/soilgrids/v2.0/properties/query
Auth: None required
Rate limit: Not explicitly documented; be respectful (~60 req/min)
Coverage: Global land surfaces, 250m resolution
Depths: 0-5cm, 5-15cm, 15-30cm, 30-60cm, 60-100cm, 100-200cm

SoilGrids provides predictions for soil properties using machine learning
trained on ~240,000 soil profile observations. Data is published under
CC-BY 4.0.

Properties:
  - bdod: Bulk density of the fine earth fraction (cg/cm³)
  - cec: Cation Exchange Capacity (mmol(c)/kg)
  - cfvo: Coarse fragments volumetric (cm³/dm³)
  - clay: Clay content (g/kg)
  - nitrogen: Total nitrogen (cg/kg)
  - ocd: Organic carbon density (hg/m³)
  - ocs: Organic carbon stocks (t/ha) — only for 0-30cm
  - phh2o: pH in H2O (pH×10)
  - sand: Sand content (g/kg)
  - silt: Silt content (g/kg)
  - soc: Soil organic carbon (dg/kg)
  - wv0010: Volumetric Water Content at 10 kPa (0.1 v% or 1 cm³/dm³)
  - wv0033: Volumetric Water Content at 33 kPa (0.1 v% or 1 cm³/dm³)
  - wv1500: Volumetric Water Content at 1500 kPa (0.1 v% or 1 cm³/dm³)

Value types: mean, Q0.05, Q0.5, Q0.95, uncertainty

Citation: Poggio, L., de Sousa, L.M., Batjes, N.H., et al. (2021).
SoilGrids 2.0: producing soil information for the globe with quantified
spatial uncertainty. SOIL, 7, 217–240.
https://doi.org/10.5194/soil-7-217-2021
"""

from __future__ import annotations

import logging
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
    http_get_with_retry,
)

logger = logging.getLogger(__name__)

SOILGRIDS_API_BASE = "https://rest.isric.org/soilgrids/v2.0"

# Valid property names
VALID_PROPERTIES = [
    "bdod", "cec", "cfvo", "clay", "nitrogen", "ocd", "ocs",
    "phh2o", "sand", "silt", "soc", "wv0010", "wv0033", "wv1500",
]

# Human-readable property descriptions with mapped units
PROPERTY_INFO = {
    "bdod": {"name": "Bulk density", "mapped_units": "cg/cm³", "conversion_factor": 100, "target_units": "kg/dm³"},
    "cec": {"name": "Cation Exchange Capacity", "mapped_units": "mmol(c)/kg", "conversion_factor": 10, "target_units": "cmol(c)/kg"},
    "cfvo": {"name": "Coarse fragments", "mapped_units": "cm³/dm³", "conversion_factor": 10, "target_units": "cm³/100cm³ (vol%)"},
    "clay": {"name": "Clay content", "mapped_units": "g/kg", "conversion_factor": 10, "target_units": "% (w)"},
    "nitrogen": {"name": "Total nitrogen", "mapped_units": "cg/kg", "conversion_factor": 100, "target_units": "g/kg"},
    "ocd": {"name": "Organic carbon density", "mapped_units": "hg/m³", "conversion_factor": 10, "target_units": "kg/m³"},
    "ocs": {"name": "Organic carbon stocks", "mapped_units": "t/ha", "conversion_factor": 10, "target_units": "kg/m²"},
    "phh2o": {"name": "pH in H2O", "mapped_units": "pH×10", "conversion_factor": 10, "target_units": "pH"},
    "sand": {"name": "Sand content", "mapped_units": "g/kg", "conversion_factor": 10, "target_units": "% (w)"},
    "silt": {"name": "Silt content", "mapped_units": "g/kg", "conversion_factor": 10, "target_units": "% (w)"},
    "soc": {"name": "Soil organic carbon", "mapped_units": "dg/kg", "conversion_factor": 10, "target_units": "g/kg"},
    "wv0010": {"name": "Water content at 10 kPa", "mapped_units": "0.1 v%", "conversion_factor": 10, "target_units": "vol%"},
    "wv0033": {"name": "Water content at 33 kPa", "mapped_units": "0.1 v%", "conversion_factor": 10, "target_units": "vol%"},
    "wv1500": {"name": "Water content at 1500 kPa", "mapped_units": "0.1 v%", "conversion_factor": 10, "target_units": "vol%"},
}

# Valid depth intervals
VALID_DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

# Valid value types
VALID_VALUES = ["mean", "Q0.05", "Q0.5", "Q0.95", "uncertainty"]

SOILGRIDS_DOI = "10.5194/soil-7-217-2021"
SOILGRIDS_CITATION = (
    "Poggio, L., de Sousa, L.M., Batjes, N.H., et al. (2021). "
    "SoilGrids 2.0: producing soil information for the globe with "
    "quantified spatial uncertainty. SOIL, 7, 217-240. "
    "https://doi.org/10.5194/soil-7-217-2021"
)


class SoilGridsAdapter(EcologicalAdapter):
    """Adapter for the ISRIC SoilGrids REST API v2.0."""

    @property
    def id(self) -> str:
        return "soilgrids"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_id="soilgrids",
            name="SoilGrids — ISRIC Global Soil Data",
            description=(
                "Global soil property predictions at 250m resolution from ISRIC. "
                "Provides clay, sand, silt, organic carbon, nitrogen, pH, bulk density, "
                "cation exchange capacity, and water content at six standard depths "
                "(0-5cm to 200cm). Based on machine learning trained on ~240,000 soil "
                "profiles. Essential for restoration ecology, agriculture, carbon "
                "accounting, and land-use planning."
            ),
            modalities=["sensor"],
            supports_location_search=True,
            supports_taxon_search=False,
            supports_date_range=False,
            supports_site_search=False,
            geographic_coverage="Global (land surfaces)",
            temporal_coverage_start=None,  # Static predictions, not time-series
            update_frequency="periodic (major releases)",
            quality_tier=1,  # Peer-reviewed, global standard
            requires_auth=False,
            rate_limit_per_minute=60,
            license="CC-BY-4.0",
            homepage_url="https://soilgrids.org",
        )

    async def query_properties(
        self,
        lat: float,
        lon: float,
        properties: list[str] | None = None,
        depths: list[str] | None = None,
        values: list[str] | None = None,
    ) -> dict:
        """
        Query SoilGrids for soil properties at a point location.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            properties: List of soil property codes (e.g. ['clay', 'sand', 'soc']).
                        Defaults to all available properties.
            depths: List of depth intervals (e.g. ['0-5cm', '5-15cm']).
                    Defaults to all six standard depths.
            values: List of value types (e.g. ['mean', 'Q0.05', 'Q0.95']).
                    Defaults to ['mean', 'uncertainty'].

        Returns:
            Raw API response dict with soil property predictions.
        """
        props = properties or ["clay", "sand", "silt", "soc", "nitrogen", "phh2o", "cec", "bdod"]
        depth_list = depths or VALID_DEPTHS
        value_list = values or ["mean", "uncertainty"]

        # Validate inputs
        for p in props:
            if p not in VALID_PROPERTIES:
                raise ValueError(
                    f"Invalid property '{p}'. Valid: {VALID_PROPERTIES}"
                )
        for d in depth_list:
            if d not in VALID_DEPTHS:
                raise ValueError(
                    f"Invalid depth '{d}'. Valid: {VALID_DEPTHS}"
                )
        for v in value_list:
            if v not in VALID_VALUES:
                raise ValueError(
                    f"Invalid value type '{v}'. Valid: {VALID_VALUES}"
                )

        # Build query params — SoilGrids uses repeated params for lists
        query_params: list[tuple[str, str]] = [
            ("lat", str(lat)),
            ("lon", str(lon)),
        ]
        for p in props:
            query_params.append(("property", p))
        for d in depth_list:
            query_params.append(("depth", d))
        for v in value_list:
            query_params.append(("value", v))

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await http_get_with_retry(
                client,
                f"{SOILGRIDS_API_BASE}/properties/query",
                params=query_params,
            )
            resp.raise_for_status()
            return resp.json()

    async def search(self, params: SearchParams) -> list[EcologicalObservation]:
        """
        Search SoilGrids data as EcologicalObservation records.

        For SoilGrids, "search" means: given a location, return soil property
        data at all standard depths. Each depth becomes one observation record.
        """
        if params.lat is None or params.lng is None:
            return []

        try:
            raw = await self.query_properties(
                lat=params.lat,
                lon=params.lng,
            )
        except Exception as exc:
            logger.warning("SoilGrids query failed: %s", exc)
            return []

        return _response_to_observations(raw, params.lat, params.lng, params.limit)

    async def get_by_id(self, source_id: str) -> Optional[EcologicalObservation]:
        """
        SoilGrids doesn't have individual record IDs.
        IDs are synthetic: 'soilgrids:{lat},{lng}:{depth}'.
        Parse the ID and fetch that location's data.
        """
        try:
            parts = source_id.split(":")
            if len(parts) < 2:
                return None
            coords = parts[0].split(",")
            lat, lng = float(coords[0]), float(coords[1])
            depth = parts[1] if len(parts) >= 2 else "0-5cm"

            raw = await self.query_properties(
                lat=lat, lon=lng, depths=[depth],
            )
            observations = _response_to_observations(raw, lat, lng, limit=1)
            return observations[0] if observations else None
        except (ValueError, IndexError, KeyError):
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _response_to_observations(
    raw: dict, lat: float, lng: float, limit: int = 20
) -> list[EcologicalObservation]:
    """
    Convert a SoilGrids API response to EcologicalObservation records.

    SoilGrids response structure:
    {
        "type": "Point",
        "geometry": {"type": "Point", "coordinates": [lon, lat, elevation]},
        "properties": {
            "layers": [
                {
                    "name": "clay",
                    "unit_measure": {...},
                    "depths": [
                        {
                            "label": "0-5cm",
                            "range": {"top_depth": 0, "bottom_depth": 5, "unit_depth": "cm"},
                            "values": {"mean": 234, "uncertainty": 45, ...}
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    }

    We group by depth: one observation per depth, with all properties as value fields.
    """
    properties = raw.get("properties", {})
    layers = properties.get("layers", [])
    geometry = raw.get("geometry", {})

    # Extract resolved coordinates
    coords = geometry.get("coordinates", [lng, lat])
    resolved_lng = coords[0] if len(coords) > 0 else lng
    resolved_lat = coords[1] if len(coords) > 1 else lat
    elevation = coords[2] if len(coords) > 2 else None

    # Collect data by depth
    depth_data: dict[str, dict] = {}
    for layer in layers:
        prop_name = layer.get("name", "unknown")
        for depth_entry in layer.get("depths", []):
            label = depth_entry.get("label", "unknown")
            values = depth_entry.get("values", {})

            if label not in depth_data:
                depth_data[label] = {
                    "range": depth_entry.get("range", {}),
                }
            depth_data[label][prop_name] = values

    # Build observations — one per depth
    results = []
    # Sort depths by top_depth for consistent ordering
    sorted_depths = sorted(
        depth_data.items(),
        key=lambda item: item[1].get("range", {}).get("top_depth", 0),
    )

    for label, data in sorted_depths:
        if len(results) >= limit:
            break

        depth_range = data.pop("range", {})
        top = depth_range.get("top_depth", 0)
        bottom = depth_range.get("bottom_depth", 0)

        # Build the value payload with all properties at this depth
        value: dict = {}
        for prop_name, prop_values in data.items():
            if isinstance(prop_values, dict):
                info = PROPERTY_INFO.get(prop_name, {})
                value[prop_name] = {
                    "values": prop_values,
                    "mapped_units": info.get("mapped_units", "unknown"),
                    "description": info.get("name", prop_name),
                }

        value["depth_label"] = label
        value["depth_top_cm"] = top
        value["depth_bottom_cm"] = bottom

        obs = EcologicalObservation(
            id=f"soilgrids:{resolved_lat},{resolved_lng}:{label}",
            modality="sensor",
            taxon=None,  # Soil data has no taxonomic subject
            location=Location(
                lat=resolved_lat,
                lng=resolved_lng,
                elevation_m=elevation,
                uncertainty_m=250.0,  # SoilGrids resolution is 250m
            ),
            observed_at=datetime(2021, 1, 1, tzinfo=timezone.utc),  # SoilGrids 2.0 release
            value=value,
            unit="mixed (see value.*.mapped_units)",
            quality=Quality(
                tier=1,
                grade="research",
                validated=True,
                confidence=0.9,
                flags=["soilgrids_v2.0", f"depth_{label}", "ml_prediction"],
            ),
            provenance=Provenance(
                source_api="soilgrids",
                source_id=f"{resolved_lat},{resolved_lng}:{label}",
                doi=SOILGRIDS_DOI,
                license="CC-BY-4.0",
                attribution=(
                    "SoilGrids 2.0 — ISRIC World Soil Information. "
                    "Global soil property predictions at 250m resolution."
                ),
                citation_string=SOILGRIDS_CITATION,
                institution_code="ISRIC",
                original_url=(
                    f"https://rest.isric.org/soilgrids/v2.0/properties/query"
                    f"?lat={resolved_lat}&lon={resolved_lng}"
                ),
            ),
            raw=raw,
        )
        results.append(obs)

    return results
