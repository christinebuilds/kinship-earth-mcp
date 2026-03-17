"""
SoilGrids MCP server — exposes ISRIC SoilGrids soil property data to AI agents
via the Model Context Protocol.

SoilGrids provides global soil property predictions at 250m resolution — essential
for restoration ecology, agriculture, carbon accounting, and land-use planning.

Tools follow MCP naming conventions:
- snake_case, prefixed with 'soilgrids_'
- Descriptions written for LLM understanding (intention-focused)

Run locally:   uv run mcp dev src/soilgrids_mcp/server.py
Run via HTTP:  uv run python -m soilgrids_mcp.server
"""

from __future__ import annotations

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from soilgrids_mcp.adapter import (
    SoilGridsAdapter,
    VALID_PROPERTIES,
    VALID_DEPTHS,
    VALID_VALUES,
    PROPERTY_INFO,
)

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "soilgrids",
    instructions=(
        "This server provides access to SoilGrids — ISRIC's global soil property "
        "predictions at 250m resolution. Use soilgrids_get_soil_properties to fetch "
        "soil data (clay, sand, silt, organic carbon, nitrogen, pH, bulk density, "
        "CEC, water content) at any point on Earth, at six depth intervals from "
        "0-5cm to 200cm. Use soilgrids_list_properties to see all available soil "
        "properties and their units. Common ecological use cases: understanding soil "
        "conditions for habitat restoration, carbon stock estimation, agricultural "
        "suitability, and correlating soil chemistry with species distributions."
    ),
)

_adapter = SoilGridsAdapter()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def soilgrids_get_soil_properties(
    lat: float,
    lon: float,
    properties: Optional[str] = None,
    depths: Optional[str] = None,
) -> dict:
    """
    Get soil property predictions from SoilGrids for a specific location.

    SoilGrids provides global soil data at 250m resolution, predicted using
    machine learning trained on ~240,000 soil profiles worldwide.

    Args:
        lat: Latitude in decimal degrees (e.g. 37.77 for San Francisco).
        lon: Longitude in decimal degrees (e.g. -122.43). Negative = West.
        properties: Comma-separated soil property codes. Defaults to core set:
                    clay, sand, silt, soc, nitrogen, phh2o, cec, bdod.
                    Use soilgrids_list_properties to see all available.
        depths: Comma-separated depth intervals. Defaults to all six:
                0-5cm, 5-15cm, 15-30cm, 30-60cm, 60-100cm, 100-200cm.

    Returns soil predictions with mean values and uncertainty bounds at each depth.
    Values are in mapped units (see property descriptions for conversion).
    """
    prop_list = None
    if properties:
        prop_list = [p.strip() for p in properties.split(",")]

    depth_list = None
    if depths:
        depth_list = [d.strip() for d in depths.split(",")]

    raw = await _adapter.query_properties(
        lat=lat, lon=lon,
        properties=prop_list,
        depths=depth_list,
        values=["mean", "Q0.05", "Q0.95", "uncertainty"],
    )

    # Extract coordinates
    geometry = raw.get("geometry", {})
    coords = geometry.get("coordinates", [lon, lat])

    return {
        "location": {
            "requested": {"lat": lat, "lon": lon},
            "resolved": {
                "lat": coords[1] if len(coords) > 1 else lat,
                "lon": coords[0] if len(coords) > 0 else lon,
                "elevation_m": coords[2] if len(coords) > 2 else None,
            },
            "note": "SoilGrids resolution is 250m. Values represent the nearest grid cell.",
        },
        "layers": raw.get("properties", {}).get("layers", []),
        "provenance": {
            "source": "SoilGrids 2.0 (ISRIC)",
            "doi": "10.5194/soil-7-217-2021",
            "license": "CC-BY-4.0",
        },
    }


@mcp.tool()
async def soilgrids_list_properties() -> dict:
    """
    List all available SoilGrids soil properties, depths, and their units.

    Returns the complete list of soil properties that can be queried,
    their mapped units, and the available depth intervals. Use these
    names with soilgrids_get_soil_properties.
    """
    return {
        "properties": {
            code: {
                "name": info["name"],
                "mapped_units": info["mapped_units"],
                "target_units": info["target_units"],
                "conversion_factor": info["conversion_factor"],
                "note": f"Divide mapped value by {info['conversion_factor']} to get {info['target_units']}",
            }
            for code, info in PROPERTY_INFO.items()
        },
        "depths": VALID_DEPTHS,
        "value_types": {
            "mean": "Predicted mean value",
            "Q0.05": "5th percentile (lower bound of 90% prediction interval)",
            "Q0.5": "Median prediction",
            "Q0.95": "95th percentile (upper bound of 90% prediction interval)",
            "uncertainty": "Width of the 90% prediction interval",
        },
        "notes": {
            "resolution": "250m global grid",
            "ocs_depth": "Organic carbon stocks (ocs) is only available for 0-30cm aggregate",
            "water_content": (
                "wv0010, wv0033, wv1500 = volumetric water content at different "
                "matric potentials (field capacity, wilting point)."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
