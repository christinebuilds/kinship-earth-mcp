"""Movebank MCP server — standalone for individual deployment."""

from mcp.server.fastmcp import FastMCP

from .adapter import MovebankAdapter

mcp = FastMCP(
    "movebank",
    instructions=(
        "Movebank provides GPS tracking data for thousands of animal species. "
        "Use this to query animal movement paths, migration routes, and habitat use patterns."
    ),
)

_adapter = MovebankAdapter()


@mcp.tool()
async def movebank_search(
    taxon: str | None = None,
    lat: float | None = None,
    lng: float | None = None,
    radius_km: float = 200,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> dict:
    """Search Movebank for animal tracking data."""
    from kinship_shared import SearchParams
    params = SearchParams(
        taxon=taxon, lat=lat, lng=lng, radius_km=radius_km,
        start_date=start_date, end_date=end_date, limit=limit,
    )
    results = await _adapter.search(params)
    return {
        "source": "movebank",
        "count": len(results),
        "observations": [r.model_dump() for r in results],
    }


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
