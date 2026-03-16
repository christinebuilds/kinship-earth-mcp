"""
OBIS MCP server — exposes marine biodiversity occurrence data to AI agents via
the Model Context Protocol.

Tools follow MCP naming conventions:
- snake_case, prefixed with 'obis_'
- Descriptions written for LLM understanding (intention-focused)

Run locally:   uv run mcp dev src/obis_mcp/server.py
Run via HTTP:  uv run python -m obis_mcp.server
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from mcp.server.fastmcp import FastMCP

from obis_mcp.adapter import OBISAdapter

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "obis",
    instructions=(
        "This server provides access to OBIS (Ocean Biodiversity Information System) data — "
        "the world's largest open repository of marine species occurrence records. "
        "168M+ records spanning 166K+ marine species: fish, marine mammals, invertebrates, "
        "algae, plankton, and more. Data is Darwin Core-native with full taxonomy. "
        "Use obis_search_occurrences to find species records by location, taxon name, or date range. "
        "Use obis_get_occurrence to retrieve a specific record by its OBIS UUID. "
        "Use obis_get_statistics to see database-wide totals."
    ),
)

_adapter = OBISAdapter()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def obis_search_occurrences(
    scientificname: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    output_format: Literal["json", "geojson"] = "json",
) -> list[dict] | dict:
    """
    Search OBIS for marine species occurrence records.

    OBIS holds 168M+ occurrence records for 166K+ marine species worldwide.
    Use this for marine/ocean species. For terrestrial biodiversity, prefer
    inaturalist_search. For cross-source queries, use ecology_search on the orchestrator.

    Args:
        scientificname: Scientific name to search for, e.g. 'Delphinus delphis'
                        (common dolphin), 'Tursiops truncatus' (bottlenose dolphin).
                        Matches the full OBIS taxonomic index.
        lat: Latitude of the center point for geographic search (decimal degrees).
        lon: Longitude of the center point (decimal degrees, negative = West).
        radius_km: Search radius in kilometres. Required if lat/lon are provided.
        start_date: Start of date range, ISO 8601 format (e.g. '2015-01-01').
        end_date: End of date range, ISO 8601 format (e.g. '2023-12-31').
        limit: Maximum number of records to return (default 20, max 500).
        offset: Number of results to skip for pagination (default 0).
        output_format: "json" (default) returns a list of dicts.
                       "geojson" returns a GeoJSON FeatureCollection.

    Example return (json): [{"id": "obis:abc-123", "taxon": {"scientific_name": "Delphinus delphis",
    "common_name": "Common Dolphin"}, "location": {"lat": 36.7, "lon": -122.0},
    "observed_at": "2023-06-15T00:00:00", "quality": {"tier": 2}}]
    """
    from kinship_shared import SearchParams, observations_to_geojson

    params = SearchParams(
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        taxon=scientificname,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )
    observations_raw = await _adapter.search(params)
    observations = [_obs_to_dict(obs) for obs in observations_raw]

    if output_format == "geojson":
        return observations_to_geojson(observations)
    return observations


@mcp.tool()
async def obis_get_occurrence(occurrence_id: str) -> Optional[dict]:
    """
    Retrieve a single OBIS occurrence record by its UUID.

    Args:
        occurrence_id: The OBIS occurrence UUID, e.g.
                       '00002b1a-306d-479b-bd91-aa7cfe4dc2ef'.
                       These IDs appear in the 'id' field of search results.

    Returns the full occurrence record with taxonomy, coordinates, date,
    and provenance. Returns null if the ID is not found.
    """
    obs = await _adapter.get_by_id(occurrence_id)
    if obs is None:
        return None
    return _obs_to_dict(obs)


@mcp.tool()
async def obis_get_statistics() -> dict:
    """
    Get OBIS database-wide statistics.

    Returns total record count, species count, dataset count, and other
    summary figures for the entire OBIS repository. Useful for understanding
    the scale and scope of the database before querying.
    """
    return await _adapter.get_statistics()


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _obs_to_dict(obs: Any) -> dict:
    """Convert an EcologicalObservation to a clean dict for MCP output."""
    taxon = obs.taxon
    loc = obs.location
    prov = obs.provenance
    qual = obs.quality

    return {
        "id": obs.id,
        "modality": obs.modality,
        "taxon": {
            "scientific_name": taxon.scientific_name,
            "common_name": taxon.common_name,
            "rank": taxon.rank,
            "kingdom": taxon.kingdom,
            "phylum": taxon.phylum,
            "class": taxon.class_,
            "order": taxon.order,
            "family": taxon.family,
        } if taxon else None,
        "location": {
            "lat": loc.lat,
            "lon": loc.lng,
            "uncertainty_m": loc.uncertainty_m,
            "country": loc.country,
            "state_province": loc.state_province,
        },
        "observed_at": obs.observed_at.isoformat(),
        "value": obs.value,
        "quality": {
            "tier": qual.tier,
            "grade": qual.grade,
            "validated": qual.validated,
            "flags": qual.flags,
        },
        "provenance": {
            "source_api": prov.source_api,
            "source_id": prov.source_id,
            "license": prov.license,
            "dataset_id": prov.dataset_id,
            "institution_code": prov.institution_code,
            "original_url": prov.original_url,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
