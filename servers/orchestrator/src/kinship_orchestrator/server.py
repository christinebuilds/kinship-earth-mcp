"""
Kinship Earth Orchestrator — cross-source ecological intelligence tools.

This is the unique value of Kinship Earth. No individual data source can
answer cross-domain questions like "what was the climate when this species
was observed?" or "what ecological data exists for this location?"

The orchestrator coordinates across NEON, OBIS, and ERA5 (and future sources)
to answer higher-level ecological questions.

Tools are unprefixed with 'ecology_' per the architecture conventions:
- ecology_get_environmental_context — climate + sensors for a point/time
- ecology_search — unified search across all sources
- ecology_describe_sources — what data is available

Run locally:   uv run mcp dev src/kinship_orchestrator/server.py
Run via HTTP:  uv run python -m kinship_orchestrator.server
"""

from __future__ import annotations

import os
from typing import Literal, Optional

from mcp.server.fastmcp import FastMCP

from neonscience_mcp.adapter import NeonAdapter
from obis_mcp.adapter import OBISAdapter
from era5_mcp.adapter import ERA5Adapter
from inaturalist_mcp.adapter import INaturalistAdapter
from ebird_mcp.adapter import EBirdAdapter

from kinship_shared import (
    run_describe_sources,
    run_get_environmental_context,
    run_search,
)

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "kinship-earth",
    instructions=(
        "This is the Kinship Earth orchestrator — the cross-source ecological intelligence "
        "layer. Use ecology_get_environmental_context to get climate, soil, and sensor data "
        "for any location and time period (combines ERA5 climate + nearest NEON sites). "
        "Use ecology_search to search for species observations AND their environmental "
        "context in a single call (combines OBIS marine data + NEON terrestrial data + ERA5 "
        "climate). Use ecology_describe_sources to learn what data sources are available "
        "and their capabilities. These tools answer questions that no single data source "
        "can answer alone."
    ),
)

# Initialize all adapters
_neon = NeonAdapter(api_token=os.environ.get("NEON_API_TOKEN"))
_obis = OBISAdapter()
_era5 = ERA5Adapter()
_inat = INaturalistAdapter()
_ebird = EBirdAdapter(api_key=os.environ.get("EBIRD_API_KEY"))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def ecology_get_environmental_context(
    lat: float,
    lon: float,
    date: str,
    days_before: int = 7,
    days_after: int = 0,
) -> dict:
    """
    Get the full environmental context for a location and time.

    This is the flagship cross-source tool. Given a point on Earth and a date,
    it pulls climate data from ERA5 and finds the nearest NEON monitoring sites.
    Use this to understand the environmental conditions when and where a species
    was observed, or to characterize a study site.

    Args:
        lat: Latitude in decimal degrees (e.g. 45.82 for Wind River).
        lon: Longitude in decimal degrees (e.g. -121.95). Negative = West.
        date: The focal date in ISO 8601 format (e.g. '2023-06-15').
        days_before: Number of days before the focal date to include in the
                     climate window (default 7). Useful for seeing conditions
                     leading up to an observation.
        days_after: Number of days after the focal date to include (default 0).

    Returns:
        - ERA5 daily climate summary for the time window
        - Nearest NEON field sites (within 200km)
        - Location metadata
    """
    return await run_get_environmental_context(
        lat=lat, lon=lon, date=date,
        days_before=days_before, days_after=days_after,
        neon=_neon, era5=_era5,
    )


@mcp.tool()
async def ecology_search(
    scientificname: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_climate: bool = True,
    limit: int = 20,
    offset: int = 0,
    output_format: Literal["json", "geojson"] = "json",
) -> dict:
    """
    Unified ecological search across all Kinship Earth data sources.

    Searches OBIS (marine species) and NEON (terrestrial sites) simultaneously,
    and optionally adds ERA5 climate context for the search area and time period.
    This is the single entry point for cross-source ecological data discovery.

    For source-specific queries with more control, use the individual servers:
    obis_search_occurrences (marine), inaturalist_search (all taxa),
    ebird_recent_observations (birds, real-time).

    Args:
        scientificname: Scientific name to search for (e.g. 'Delphinus delphis').
                        Searches OBIS occurrence records.
        lat: Latitude for geographic search (decimal degrees).
        lon: Longitude for geographic search (negative = West).
        radius_km: Search radius in kilometres.
        start_date: Start of date range (ISO 8601, e.g. '2020-01-01').
        end_date: End of date range (ISO 8601, e.g. '2023-12-31').
        include_climate: If true and lat/lon/dates are provided, include ERA5
                         daily climate summary for the search area (default true).
        limit: Max records per source (default 20).
        offset: Number of results to skip for pagination (default 0).
        output_format: "json" (default) returns structured dict with sections.
                       "geojson" returns species_occurrences as a GeoJSON
                       FeatureCollection (useful for QGIS and mapping tools).

    Example return (json): {"species_occurrences": [...], "neon_sites": [...],
    "climate": {"daily": [...]}, "search_context": {"relevance_scores": [...]}}
    """
    result = await run_search(
        scientificname=scientificname, lat=lat, lon=lon,
        radius_km=radius_km, start_date=start_date, end_date=end_date,
        include_climate=include_climate, limit=limit,
        obis=_obis, neon=_neon, era5=_era5,
        inat=_inat, ebird=_ebird,
    )

    if output_format == "geojson":
        from kinship_shared import observations_to_geojson
        occurrences = result.get("species_occurrences", [])
        result["species_occurrences_geojson"] = observations_to_geojson(occurrences)

    return result


@mcp.tool()
async def ecology_describe_sources() -> dict:
    """
    Describe all available ecological data sources and their capabilities.

    Returns a summary of each data source including what types of data it
    provides, geographic and temporal coverage, quality tier, and access
    requirements. Use this to understand what data is available before
    searching.
    """
    return await run_describe_sources(neon=_neon, obis=_obis, era5=_era5, inat=_inat, ebird=_ebird)


# DRAFT: citizen-facing discovery tool (pending full eBird + iNaturalist validation)
@mcp.tool()
async def ecology_whats_around_me(
    lat: float,
    lon: float,
    radius_km: float = 25,
    days_back: int = 7,
) -> dict:
    """
    Discover what's been observed near a location recently.

    A citizen-friendly entry point: "What's happening in the ecosystem around me?"
    Returns a snapshot combining recent species sightings, nearby monitoring sites,
    and current climate conditions from all available data sources.

    Args:
        lat: Your latitude in decimal degrees.
        lon: Your longitude in decimal degrees (negative = West).
        radius_km: How far to look (default 25 km).
        days_back: How many days of recent observations to include (default 7).

    Returns a combined ecological snapshot: recent species sightings (from eBird,
    iNaturalist, and OBIS), nearby monitoring sites (NEON), and current climate
    conditions (ERA5).
    """
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    result = await run_search(
        lat=lat, lon=lon,
        radius_km=radius_km,
        start_date=start_date,
        end_date=end_date,
        include_climate=True,
        limit=30,
        obis=_obis, neon=_neon, era5=_era5,
        inat=_inat, ebird=_ebird,
    )

    # Summarize by source for the citizen
    by_source = {}
    for occ in result.get("species_occurrences", []):
        src = occ.get("source", "obis")
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(occ)

    # Count unique species
    species_names = set()
    for occ in result.get("species_occurrences", []):
        name = occ.get("scientific_name")
        if name:
            species_names.add(name)

    return {
        "snapshot": {
            "location": {"lat": lat, "lon": lon, "radius_km": radius_km},
            "period": f"Last {days_back} days ({start_date} to {end_date})",
            "unique_species": len(species_names),
            "total_observations": len(result.get("species_occurrences", [])),
            "monitoring_sites_nearby": result.get("neon_site_count", 0),
            "climate_data_available": result.get("climate") is not None,
        },
        "recent_sightings": result.get("species_occurrences", [])[:20],
        "neon_sites": result.get("neon_sites", []),
        "climate": result.get("climate"),
        "sources_queried": result.get("search_context", {}).get("sources_queried", []),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
