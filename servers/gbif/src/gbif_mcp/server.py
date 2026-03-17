"""
GBIF MCP Server — Global Biodiversity Information Facility.

Exposes GBIF's 2.8B+ occurrence records as MCP tools.
No authentication required.

Run: uv run mcp dev src/gbif_mcp/server.py
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from .adapter import GBIFAdapter

mcp = FastMCP(
    "gbif",
    instructions=(
        "GBIF is the world's largest biodiversity data aggregator with 2.8B+ "
        "occurrence records from 90,000+ datasets. Use gbif_search_occurrences "
        "to find species observation records by name, location, or date range. "
        "GBIF aggregates data from eBird, iNaturalist, museums, herbaria, and "
        "national monitoring programs worldwide."
    ),
)

_adapter = GBIFAdapter()


@mcp.tool()
async def gbif_search_occurrences(
    scientificname: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """
    Search GBIF for species occurrence records.

    GBIF aggregates 2.8B+ records from museums, herbaria, citizen science
    (eBird, iNaturalist), and monitoring programs worldwide.

    Args:
        scientificname: Species name (e.g. 'Quercus alba')
        lat: Latitude for geographic search
        lon: Longitude (negative = West)
        radius_km: Search radius in km (default 200)
        start_date: Start date (ISO 8601)
        end_date: End date (ISO 8601)
        limit: Max records (default 20, max 300)
    """
    from kinship_shared import SearchParams

    params = SearchParams(
        taxon=scientificname,
        lat=lat,
        lng=lon,
        radius_km=radius_km,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    observations = await _adapter.search(params)
    return [
        {
            "id": obs.id,
            "scientific_name": obs.taxon.scientific_name if obs.taxon else None,
            "lat": obs.location.lat,
            "lng": obs.location.lng,
            "country": obs.location.country,
            "observed_at": obs.observed_at.isoformat(),
            "basis_of_record": obs.value.get("basis_of_record") if obs.value else None,
            "institution": obs.value.get("institution") if obs.value else None,
            "quality_tier": obs.quality.tier,
            "source_url": obs.provenance.original_url,
            "license": obs.provenance.license,
        }
        for obs in observations
    ]


@mcp.tool()
async def gbif_get_occurrence(gbif_key: str) -> dict | None:
    """
    Fetch a specific GBIF occurrence record by its key.

    Args:
        gbif_key: The GBIF occurrence key (numeric string)
    """
    obs = await _adapter.get_by_id(gbif_key)
    if obs is None:
        return None
    return {
        "id": obs.id,
        "scientific_name": obs.taxon.scientific_name if obs.taxon else None,
        "taxonomy": {
            "kingdom": obs.taxon.kingdom if obs.taxon else None,
            "phylum": obs.taxon.phylum if obs.taxon else None,
            "family": obs.taxon.family if obs.taxon else None,
            "genus": obs.taxon.genus if obs.taxon else None,
        },
        "lat": obs.location.lat,
        "lng": obs.location.lng,
        "country": obs.location.country,
        "observed_at": obs.observed_at.isoformat(),
        "value": obs.value,
        "quality": {"tier": obs.quality.tier, "grade": obs.quality.grade, "flags": obs.quality.flags},
        "provenance": {
            "source_url": obs.provenance.original_url,
            "license": obs.provenance.license,
            "dataset_id": obs.provenance.dataset_id,
            "institution": obs.provenance.institution_code,
            "citation": obs.provenance.citation_string,
        },
    }


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
