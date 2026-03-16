"""
USGS NWIS MCP Server — stream gauge and water quality tools for AI agents.

Exposes USGS water monitoring data through the Model Context Protocol.
No API key required (but recommended for higher rate limits).
"""

import os

from mcp.server.fastmcp import FastMCP

from usgs_nwis_mcp.adapter import USGSNWISAdapter

mcp = FastMCP(
    "usgs-nwis",
    instructions=(
        "USGS National Water Information System provides real-time stream gauge "
        "data from ~13,500 stations across the United States. Use this server to "
        "check streamflow, water temperature, dissolved oxygen, pH, and other "
        "water quality parameters. Data is updated every 15 minutes. This is the "
        "primary source for watershed health monitoring in the US."
    ),
)

_adapter = USGSNWISAdapter(api_key=os.environ.get("USGS_API_KEY"))


def _obs_to_dict(obs) -> dict:
    """Serialize an EcologicalObservation to a clean dict for MCP output."""
    return {
        "id": obs.id,
        "site_id": obs.location.site_id,
        "site_name": obs.location.site_name,
        "lat": obs.location.lat,
        "lng": obs.location.lng,
        "state": obs.location.state_province,
        "watershed_id": obs.location.watershed_id,
        "observed_at": obs.observed_at.isoformat(),
        "parameter": obs.value.get("parameter_name") if obs.value else None,
        "value": obs.value.get("measurement") if obs.value else None,
        "unit": obs.unit,
        "temporal_resolution": obs.temporal_resolution,
        "approved": obs.quality.validated,
        "site_url": obs.provenance.original_url,
    }


@mcp.tool()
async def usgs_stream_conditions(
    lat: float,
    lon: float,
    radius_km: float = 50,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    Get stream gauge data near a location.

    Returns recent streamflow, water temperature, and other measurements
    from USGS monitoring stations. Use this to understand watershed
    conditions, check stream health, or monitor for drought/flood.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees (negative = West)
        radius_km: Search radius in km (max 200)
        start_date: Start date in YYYY-MM-DD format (optional, defaults to recent)
        end_date: End date in YYYY-MM-DD format (optional)
        limit: Max results to return (default 20)
    """
    from kinship_shared import SearchParams

    params = SearchParams(
        lat=lat,
        lng=lon,
        radius_km=radius_km,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    results = await _adapter.search(params)
    return [_obs_to_dict(obs) for obs in results]


@mcp.tool()
async def usgs_site_data(
    site_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Get data from a specific USGS monitoring station.

    Args:
        site_id: USGS site number (e.g., '01646500' for Potomac River).
                 The 'USGS-' prefix is added automatically if not present.
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        limit: Max results (default 50)
    """
    from kinship_shared import SearchParams

    params = SearchParams(
        site_id=site_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    results = await _adapter.search(params)
    return [_obs_to_dict(obs) for obs in results]
