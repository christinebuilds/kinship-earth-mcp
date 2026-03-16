"""
NEON MCP server — exposes NEON ecological data to AI agents via the
Model Context Protocol.

Tools follow MCP naming conventions (SEP-986):
- snake_case, prefixed with 'neon_'
- Descriptions written for LLM understanding (intention-focused)

Run locally:   uv run mcp dev src/neonscience_mcp/server.py
Run via HTTP:  uv run python -m neonscience_mcp.server
"""

from __future__ import annotations

import os
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from neonscience_mcp.adapter import NeonAdapter

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "neonscience",
    instructions=(
        "This server provides access to NEON (National Ecological Observatory Network) data. "
        "NEON operates 81 field sites across the United States, collecting standardized "
        "ecological data including soil sensors, bird surveys, aquatic monitoring, "
        "eddy covariance flux measurements, airborne imagery, and bioacoustics. "
        "All NEON data is research-grade (quality tier 1) and freely accessible. "
        "Use neon_list_sites to discover available field sites, "
        "neon_get_site for details on a specific site, "
        "neon_list_data_products to see what types of data are available, and "
        "neon_search_observations to find observations near a location or at a named site."
    ),
)

_adapter = NeonAdapter(api_token=os.environ.get("NEON_API_TOKEN"))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def neon_list_sites() -> list[dict]:
    """
    List all 81 NEON field sites with their location, domain, and state.

    Returns a summary of each site including: site code (e.g. 'WREF'),
    site name, latitude, longitude, site type (CORE or RELOCATABLE),
    ecoclimatic domain, and US state. Use this to discover what sites
    exist before querying for specific data.
    """
    sites = await _adapter.list_sites()
    # Return a clean summary rather than the full raw response
    return [
        {
            "site_code": s.get("siteCode"),
            "site_name": s.get("siteName"),
            "site_type": s.get("siteType"),
            "domain_code": s.get("domainCode"),
            "domain_name": s.get("domainName"),
            "state": s.get("stateCode"),
            "lat": s.get("siteLatitude"),
            "lon": s.get("siteLongitude"),
            "elevation_m": s.get("siteElevation"),
            "data_products_available": len(s.get("dataProducts", [])),
        }
        for s in sites
        if s.get("siteCode")
    ]


@mcp.tool()
async def neon_get_site(site_code: str) -> Optional[dict]:
    """
    Get full details for a specific NEON field site by its code.

    Args:
        site_code: The NEON site code, e.g. 'WREF' (Wind River Experimental Forest),
                   'HARV' (Harvard Forest), 'KONZ' (Konza Prairie). Case-insensitive.

    Returns full site details including location, ecological domain, available
    data products, and operational dates. Returns null if the site code is not found.
    """
    site = await _adapter.get_site(site_code)
    if not site:
        return None
    return {
        "site_code": site.get("siteCode"),
        "site_name": site.get("siteName"),
        "site_type": site.get("siteType"),
        "site_description": site.get("siteDescription"),
        "domain_code": site.get("domainCode"),
        "domain_name": site.get("domainName"),
        "state": site.get("stateCode"),
        "country": "United States",
        "lat": site.get("siteLatitude"),
        "lon": site.get("siteLongitude"),
        "elevation_m": site.get("siteElevation"),
        "data_products": [
            {"product_code": dp.get("dataProductCode"), "product_name": dp.get("dataProductTitle")}
            for dp in site.get("dataProducts", [])
        ],
        "portal_url": f"https://www.neonscience.org/field-sites/{site.get('siteCode', '').lower()}",
    }


@mcp.tool()
async def neon_list_data_products(keyword: Optional[str] = None) -> list[dict]:
    """
    List available NEON data products, optionally filtered by keyword.

    NEON provides 180+ standardized data products covering soil biogeochemistry,
    bird point counts, aquatic macroinvertebrates, eddy covariance flux,
    airborne hyperspectral imagery, lidar, and more.

    Args:
        keyword: Optional filter. Search term to match against product names
                 or descriptions. E.g. 'bird', 'soil', 'temperature', 'acoustic'.

    Returns product code, name, and description for each matching product.
    """
    products = await _adapter.list_data_products()
    results = []
    kw = keyword.lower() if keyword else None

    for p in products:
        name = p.get("productName", "")
        desc = p.get("productDescription", "")
        if kw and kw not in name.lower() and kw not in desc.lower():
            continue
        results.append({
            "product_code": p.get("productCode"),
            "product_name": name,
            "product_description": desc,
            "product_status": p.get("productStatus"),
            "science_team": p.get("productScienceTeam"),
        })

    return results


@mcp.tool()
async def neon_search_observations(
    site_code: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = 100.0,
) -> list[dict]:
    """
    Search for NEON field sites near a location or by site code.

    Use this to find which NEON sites have data available for a geographic area
    or to get observation context for a specific named site.

    Args:
        site_code: NEON site code to look up directly (e.g. 'WREF', 'HARV').
        lat: Latitude of the center point for geographic search.
        lon: Longitude of the center point for geographic search (negative = West).
        radius_km: Search radius in kilometres (default 100km). Used with lat/lon.

    Returns site information as EcologicalObservation-compatible records.
    """
    from kinship_shared import SearchParams

    params = SearchParams(
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        site_id=site_code,
        limit=50,
    )
    observations = await _adapter.search(params)
    return [
        {
            "id": obs.id,
            "site_code": obs.location.site_id,
            "site_name": obs.location.site_name,
            "lat": obs.location.lat,
            "lon": obs.location.lng,
            "elevation_m": obs.location.elevation_m,
            "state": obs.location.state_province,
            "modality": obs.modality,
            "quality_tier": obs.quality.tier,
            "data_products_available": obs.value.get("data_products_available") if obs.value else None,
            "portal_url": obs.provenance.original_url,
        }
        for obs in observations
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
