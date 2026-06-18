"""FLUXNET MCP server — standalone for individual deployment."""

from mcp.server.fastmcp import FastMCP

from .adapter import FLUXNETAdapter

mcp = FastMCP(
    "fluxnet",
    instructions=(
        "FLUXNET provides carbon, water, and energy flux measurements from "
        "eddy covariance towers worldwide. Use this to query ecosystem "
        "carbon budgets, primary productivity, and energy balance data."
    ),
)

_adapter = FLUXNETAdapter()


@mcp.tool()
async def fluxnet_search(
    lat: float | None = None,
    lng: float | None = None,
    radius_km: float = 200,
    site_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> dict:
    """Search for FLUXNET / AmeriFlux tower data."""
    from kinship_shared import SearchParams
    params = SearchParams(
        lat=lat, lng=lng, radius_km=radius_km, site_id=site_id,
        start_date=start_date, end_date=end_date, limit=limit,
    )
    results = await _adapter.search(params)
    return {
        "source": "fluxnet",
        "count": len(results),
        "observations": [r.model_dump() for r in results],
    }


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
