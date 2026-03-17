"""
iNaturalist MCP Server — biodiversity observation tools for AI agents.

Exposes iNaturalist data through the Model Context Protocol, enabling
Claude and other AI agents to search for biodiversity observations
across all taxa (plants, animals, fungi).

No API key required for read-only access.
"""

from typing import Literal

from mcp.server.fastmcp import FastMCP

from inaturalist_mcp.adapter import INaturalistAdapter

mcp = FastMCP(
    "inaturalist",
    instructions=(
        "iNaturalist is a global community biodiversity platform with over "
        "200 million observations of plants, animals, fungi, and other "
        "organisms. Observations are photo-verified through community "
        "consensus. Use this server to search for species observations "
        "by location, taxon, and date range. Research-grade observations "
        "have been validated by multiple community identifiers."
    ),
)

_adapter = INaturalistAdapter()


def _obs_to_dict(obs) -> dict:
    """Serialize an EcologicalObservation to a clean dict for MCP output."""
    result = {
        "id": obs.id,
        "scientific_name": obs.taxon.scientific_name if obs.taxon else None,
        "common_name": obs.taxon.common_name if obs.taxon else None,
        "lat": obs.location.lat,
        "lon": obs.location.lng,
        "place": obs.location.country,
        "observed_at": obs.observed_at.isoformat(),
        "quality_grade": obs.quality.grade,
        "photo_url": obs.media_url,
        "observation_url": obs.provenance.original_url,
        "license": obs.provenance.license,
    }
    if obs.value:
        result["identifications_count"] = obs.value.get("identifications_count", 0)
        result["iconic_taxon"] = obs.value.get("iconic_taxon")
    return result


@mcp.tool()
async def inaturalist_search(
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 25,
    taxon_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    quality: str = "verifiable",
    limit: int = 20,
    offset: int = 0,
    output_format: Literal["json", "geojson"] = "json",
) -> list[dict] | dict:
    """
    Search for biodiversity observations on iNaturalist.

    Covers all taxa: plants, animals, fungi, insects, and more. Use this for
    community-verified biodiversity surveys. For marine-only species, prefer
    obis_search_occurrences. For birds with real-time data, prefer ebird_recent_observations.

    Args:
        lat: Latitude in decimal degrees (optional)
        lon: Longitude in decimal degrees (optional, negative = West)
        radius_km: Search radius in km (max 500)
        taxon_name: Scientific or common name to search for (optional)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        quality: Quality filter — "research" for research-grade only,
                 "verifiable" for all verifiable (default)
        limit: Max results to return (default 20)
        offset: Number of results to skip for pagination (default 0)
        output_format: "json" (default) returns a list of dicts.
                       "geojson" returns a GeoJSON FeatureCollection.

    Example return (json): [{"id": "inat:12345", "scientific_name": "Quercus agrifolia",
    "common_name": "Coast Live Oak", "lat": 37.77, "lon": -122.42,
    "quality_grade": "research", "photo_url": "https://..."}]
    """
    from kinship_shared import SearchParams, observations_to_geojson

    quality_tier_min = 1 if quality == "research" else None

    params = SearchParams(
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        taxon=taxon_name,
        start_date=start_date,
        end_date=end_date,
        quality_tier_min=quality_tier_min,
        limit=limit,
        offset=offset,
    )
    results = await _adapter.search(params)
    observations = [_obs_to_dict(obs) for obs in results]

    if output_format == "geojson":
        return observations_to_geojson(observations)
    return observations


@mcp.tool()
async def inaturalist_get_observation(observation_id: str) -> dict | None:
    """
    Fetch a specific iNaturalist observation by ID.

    Args:
        observation_id: The iNaturalist observation ID (numeric string)

    Returns the observation details, or None if not found.
    """
    obs = await _adapter.get_by_id(observation_id)
    return _obs_to_dict(obs) if obs else None
