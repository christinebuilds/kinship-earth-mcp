"""
Xeno-canto MCP Server — Bird and Wildlife Sound Recordings.

Exposes Xeno-canto's 1M+ audio recordings as MCP tools.
No authentication required.

Run: uv run mcp dev src/xenocanto_mcp/server.py
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from .adapter import XenoCantoAdapter

mcp = FastMCP(
    "xeno-canto",
    instructions=(
        "Xeno-canto has 1M+ bird and wildlife sound recordings from around "
        "the world. Use xenocanto_search_recordings to find audio by species "
        "name or location. Each recording includes an audio URL, sonogram, "
        "quality rating (A-E), and recordist attribution."
    ),
)

_adapter = XenoCantoAdapter()


@mcp.tool()
async def xenocanto_search_recordings(
    scientificname: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    limit: int = 20,
) -> list[dict]:
    """
    Search Xeno-canto for bird and wildlife sound recordings.

    Args:
        scientificname: Species name (e.g. 'Turdus migratorius')
        lat: Latitude for geographic search
        lon: Longitude (negative = West)
        radius_km: Search radius in km (default 100)
        limit: Max recordings (default 20)
    """
    from kinship_shared import SearchParams

    params = SearchParams(
        taxon=scientificname,
        lat=lat,
        lng=lon,
        radius_km=radius_km,
        limit=limit,
    )
    observations = await _adapter.search(params)
    return [
        {
            "id": obs.id,
            "scientific_name": obs.taxon.scientific_name if obs.taxon else None,
            "common_name": obs.taxon.common_name if obs.taxon else None,
            "lat": obs.location.lat,
            "lng": obs.location.lng,
            "country": obs.location.country,
            "location_name": obs.location.site_name,
            "observed_at": obs.observed_at.isoformat(),
            "audio_url": obs.media_url,
            "recording_type": obs.value.get("recording_type") if obs.value else None,
            "duration": obs.value.get("duration") if obs.value else None,
            "recordist": obs.value.get("recordist") if obs.value else None,
            "quality_rating": obs.quality.flags[0] if obs.quality.flags else None,
            "quality_tier": obs.quality.tier,
            "source_url": obs.provenance.original_url,
            "license": obs.provenance.license,
            "citation": obs.provenance.citation_string,
        }
        for obs in observations
    ]


@mcp.tool()
async def xenocanto_get_recording(xc_id: str) -> dict | None:
    """
    Fetch a specific Xeno-canto recording by ID.

    Args:
        xc_id: The Xeno-canto recording ID (numeric string, e.g. '12345')
    """
    obs = await _adapter.get_by_id(xc_id)
    if obs is None:
        return None
    return {
        "id": obs.id,
        "scientific_name": obs.taxon.scientific_name if obs.taxon else None,
        "common_name": obs.taxon.common_name if obs.taxon else None,
        "lat": obs.location.lat,
        "lng": obs.location.lng,
        "country": obs.location.country,
        "audio_url": obs.media_url,
        "recording_type": obs.value.get("recording_type") if obs.value else None,
        "duration": obs.value.get("duration") if obs.value else None,
        "recordist": obs.value.get("recordist") if obs.value else None,
        "remarks": obs.value.get("remarks") if obs.value else None,
        "quality": {"tier": obs.quality.tier, "rating": obs.quality.flags[0] if obs.quality.flags else None},
        "provenance": {
            "source_url": obs.provenance.original_url,
            "license": obs.provenance.license,
            "citation": obs.provenance.citation_string,
        },
    }


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
