#!/usr/bin/env python3
"""
Fetch ecological data from Kinship Earth MCP and save as GeoJSON for QGIS.

Usage:
    python fetch_ecological_data.py --lat 36.6 --lon -121.9 --radius 50
    python fetch_ecological_data.py --lat 36.6 --lon -121.9 --species "Megaptera novaeangliae"
    python fetch_ecological_data.py --lat 36.6 --lon -121.9 --output my-observations.geojson

Requires:
    pip install httpx

The script connects to Kinship Earth's HTTP endpoint (default: http://localhost:8000)
and queries ecological data, then converts the results to a GeoJSON file ready for
QGIS import.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Error: httpx is required. Install it with: pip install httpx")
    sys.exit(1)


DEFAULT_BASE_URL = "http://localhost:8000"


def call_mcp_tool(base_url: str, tool_name: str, arguments: dict) -> dict:
    """Call a Kinship Earth MCP tool via HTTP.

    Uses the MCP Streamable HTTP transport endpoint.
    """
    url = f"{base_url}/mcp"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }
    response = httpx.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    result = response.json()
    if "error" in result:
        raise RuntimeError(f"MCP error: {result['error']}")
    return result.get("result", {})


def extract_observations(mcp_result: dict) -> list[dict]:
    """Extract observation records from an MCP tool response.

    Handles the nested content structure returned by MCP tools.
    """
    observations = []

    # MCP tools return content as a list of content blocks
    content = mcp_result.get("content", [])
    for block in content:
        if block.get("type") == "text":
            try:
                data = json.loads(block["text"])
            except (json.JSONDecodeError, KeyError):
                continue

            # ecology_search returns nested structure
            if "species_occurrences" in data:
                observations.extend(data["species_occurrences"])
            # Direct search tools return lists
            elif isinstance(data, list):
                observations.extend(data)

    return observations


def observations_to_geojson(observations: list[dict]) -> dict:
    """Convert observation records to a GeoJSON FeatureCollection.

    Handles both `lon`/`lat` and `lng`/`lat` coordinate naming conventions
    used across different Kinship Earth sub-servers.
    """
    features = []

    for obs in observations:
        # Handle coordinate naming inconsistency (lon vs lng)
        lon = obs.get("lon") or obs.get("lng") or obs.get("longitude")
        lat = obs.get("lat") or obs.get("latitude")

        if lon is None or lat is None:
            continue

        # Build properties from all non-coordinate fields
        properties = {}
        skip_keys = {"lon", "lng", "longitude", "lat", "latitude", "geometry"}
        for key, value in obs.items():
            if key not in skip_keys:
                properties[key] = value

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)],
            },
            "properties": properties,
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def fetch_ecology_search(
    base_url: str,
    lat: float,
    lon: float,
    radius_km: float,
    species: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Fetch data using the unified ecology_search tool."""
    args = {
        "lat": lat,
        "lon": lon,
        "radius_km": radius_km,
        "limit": limit,
        "include_climate": False,
    }
    if species:
        args["scientificname"] = species
    if start_date:
        args["start_date"] = start_date
    if end_date:
        args["end_date"] = end_date

    print(f"Querying ecology_search at ({lat}, {lon}), radius={radius_km}km...")
    result = call_mcp_tool(base_url, "ecology_search", args)
    return extract_observations(result)


def fetch_inaturalist(
    base_url: str,
    lat: float,
    lon: float,
    radius_km: float,
    species: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Fetch data using the iNaturalist search tool."""
    args = {
        "lat": lat,
        "lng": lon,
        "radius_km": radius_km,
        "limit": limit,
    }
    if species:
        args["taxon_name"] = species

    print(f"Querying iNaturalist at ({lat}, {lon}), radius={radius_km}km...")
    result = call_mcp_tool(base_url, "inaturalist_search", args)
    return extract_observations(result)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch ecological data from Kinship Earth MCP and save as GeoJSON.",
        epilog="Example: python fetch_ecological_data.py --lat 36.6 --lon -121.9 --radius 50",
    )
    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Latitude of the search center (e.g., 36.6 for Monterey Bay)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Longitude of the search center (e.g., -121.9 for Monterey Bay). Negative = West.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=25.0,
        help="Search radius in kilometers (default: 25)",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Filter by scientific name (e.g., 'Megaptera novaeangliae')",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of results (default: 50)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["ecology_search", "inaturalist"],
        default="ecology_search",
        help="Data source to query (default: ecology_search, which combines OBIS + NEON + ERA5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: observations-{lat}-{lon}.geojson)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Kinship Earth MCP HTTP endpoint (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the GeoJSON output",
    )

    args = parser.parse_args()

    # Determine output filename
    if args.output:
        output_path = Path(args.output)
    else:
        lat_str = f"{args.lat:.2f}".replace("-", "n")
        lon_str = f"{abs(args.lon):.2f}w" if args.lon < 0 else f"{args.lon:.2f}e"
        output_path = Path(f"observations-{lat_str}-{lon_str}.geojson")

    # Fetch data
    try:
        if args.source == "inaturalist":
            observations = fetch_inaturalist(
                args.url, args.lat, args.lon, args.radius, args.species, args.limit
            )
        else:
            observations = fetch_ecology_search(
                args.url,
                args.lat,
                args.lon,
                args.radius,
                args.species,
                args.start_date,
                args.end_date,
                args.limit,
            )
    except httpx.ConnectError:
        print(
            f"\nError: Could not connect to Kinship Earth at {args.url}",
            file=sys.stderr,
        )
        print(
            "Make sure the server is running. Start it with:",
            file=sys.stderr,
        )
        print(
            "  cd kinship-earth-mcp/servers/orchestrator && uv run main.py --transport http",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"\nError fetching data: {e}", file=sys.stderr)
        sys.exit(1)

    if not observations:
        print("No observations found. Try a larger radius or different location.")
        sys.exit(0)

    # Convert to GeoJSON
    geojson = observations_to_geojson(observations)
    feature_count = len(geojson["features"])

    # Write output
    indent = 2 if args.pretty else None
    output_path.write_text(json.dumps(geojson, indent=indent, default=str))

    print(f"\nSaved {feature_count} observations to {output_path}")
    print(f"\nTo load in QGIS:")
    print(f"  Layer > Add Layer > Add Vector Layer > {output_path}")
    print(f"  Or drag and drop the file onto the QGIS map canvas")


if __name__ == "__main__":
    main()
