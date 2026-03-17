"""GeoJSON conversion utilities for Kinship Earth MCP outputs.

Converts observation dicts to GeoJSON FeatureCollections for use in
QGIS, Jupyter mapping libraries, and other geospatial tools.
"""

from __future__ import annotations

from typing import Any


def observations_to_geojson(observations: list[dict]) -> dict:
    """Convert a list of observation dicts to a GeoJSON FeatureCollection.

    Expects each observation dict to have 'lat' and 'lng' (or 'lon') at the
    top level, OR a nested 'location' dict with 'lat' and 'lng'/'lon'.
    All other fields become Feature properties.
    """
    features = []
    for obs in observations:
        lat, lon = _extract_coords(obs)
        if lat is None or lon is None:
            continue

        # Build properties from everything except coordinates
        properties = _extract_properties(obs)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],  # GeoJSON is [lon, lat]
            },
            "properties": properties,
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _extract_coords(obs: dict) -> tuple[float | None, float | None]:
    """Extract lat/lon from an observation dict."""
    # Try top-level
    lat = obs.get("lat")
    lon = obs.get("lon") or obs.get("lng")
    if lat is not None and lon is not None:
        return lat, lon

    # Try nested location
    loc = obs.get("location")
    if isinstance(loc, dict):
        lat = loc.get("lat")
        lon = loc.get("lon") or loc.get("lng")
        return lat, lon

    return None, None


def _extract_properties(obs: dict) -> dict:
    """Extract all non-coordinate fields as GeoJSON properties."""
    skip_keys = {"lat", "lng", "lon"}
    properties: dict[str, Any] = {}

    for key, value in obs.items():
        if key in skip_keys:
            continue
        if key == "location" and isinstance(value, dict):
            # Flatten location, skip coords
            for lk, lv in value.items():
                if lk not in skip_keys:
                    properties[f"location_{lk}"] = lv
        elif isinstance(value, dict):
            # Keep nested dicts as-is (taxon, quality, provenance)
            properties[key] = value
        else:
            properties[key] = value

    return properties
