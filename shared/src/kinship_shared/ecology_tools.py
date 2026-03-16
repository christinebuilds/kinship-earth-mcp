"""
Shared tool execution logic for Kinship Earth ecology tools.

Used by both the orchestrator (MCP server) and webapp (FastAPI) to avoid
duplicating the core search/context/describe logic. Each consumer initializes
its own adapter instances and passes them in.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from .ranking import score_observation
from .schema import SearchParams

logger = logging.getLogger(__name__)


async def run_get_environmental_context(
    *,
    lat: float,
    lon: float,
    date: str,
    days_before: int = 7,
    days_after: int = 0,
    neon,
    era5,
) -> dict:
    """
    Get the full environmental context for a location and time.

    Combines ERA5 climate data with nearest NEON monitoring sites.
    """
    logger.info(
        "ecology_get_environmental_context: lat=%.4f, lon=%.4f, date=%s",
        lat, lon, date,
    )

    if not (-90 <= lat <= 90):
        return {"error": f"lat must be between -90 and 90, got {lat}"}
    if not (-180 <= lon <= 180):
        return {"error": f"lon must be between -180 and 180, got {lon}"}
    try:
        focal = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"error": f"date must be in YYYY-MM-DD format, got {date!r}"}

    start = (focal - timedelta(days=days_before)).strftime("%Y-%m-%d")
    end = (focal + timedelta(days=days_after)).strftime("%Y-%m-%d")

    era5_raw, neon_sites = await asyncio.gather(
        era5.get_daily(lat=lat, lng=lon, start_date=start, end_date=end),
        neon.search(SearchParams(lat=lat, lng=lon, radius_km=200, limit=10)),
    )
    logger.info(
        "environmental_context: %d NEON sites found, ERA5 keys=%s",
        len(neon_sites),
        list(era5_raw.get("daily", {}).keys())[:3],
    )

    nearby_neon = [
        {
            "site_code": obs.location.site_id,
            "site_name": obs.location.site_name,
            "lat": obs.location.lat,
            "lng": obs.location.lng,
            "elevation_m": obs.location.elevation_m,
            "state": obs.location.state_province,
            "data_products": obs.value.get("data_products_available") if obs.value else None,
            "portal_url": obs.provenance.original_url,
        }
        for obs in neon_sites
    ]

    return {
        "query": {
            "lat": lat,
            "lon": lon,
            "focal_date": date,
            "climate_window": {"start": start, "end": end},
        },
        "climate": {
            "source": "ERA5 (ECMWF) via Open-Meteo",
            "resolution": "~25km grid, daily aggregation",
            "location_resolved": {
                "lat": era5_raw.get("latitude"),
                "lon": era5_raw.get("longitude"),
                "elevation_m": era5_raw.get("elevation"),
            },
            "daily": era5_raw.get("daily", {}),
            "units": era5_raw.get("daily_units", {}),
            "provenance": {
                "doi": "10.24381/cds.adbb2d47",
                "license": "CC-BY-4.0",
            },
        },
        "nearby_neon_sites": nearby_neon,
        "nearby_neon_count": len(nearby_neon),
        "data_sources_used": ["era5", "neonscience"],
    }


async def run_search(
    *,
    scientificname: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_climate: bool = True,
    limit: int = 20,
    obis,
    neon,
    era5,
    inat=None,
    ebird=None,
) -> dict:
    """
    Unified ecological search across all Kinship Earth data sources.

    Searches OBIS (marine species) and NEON (terrestrial sites) simultaneously,
    and optionally adds ERA5 climate context.
    """
    logger.info(
        "ecology_search: taxon=%s, lat=%s, lon=%s, radius_km=%s",
        scientificname, lat, lon, radius_km,
    )

    if lat is not None and not (-90 <= lat <= 90):
        return {"error": f"lat must be between -90 and 90, got {lat}"}
    if lon is not None and not (-180 <= lon <= 180):
        return {"error": f"lon must be between -180 and 180, got {lon}"}

    tasks = {}

    if scientificname or (lat is not None and lon is not None):
        tasks["obis"] = obis.search(SearchParams(
            taxon=scientificname, lat=lat, lng=lon, radius_km=radius_km,
            start_date=start_date, end_date=end_date, limit=limit,
        ))

    if lat is not None and lon is not None:
        tasks["neon"] = neon.search(SearchParams(
            lat=lat, lng=lon, radius_km=radius_km or 200, limit=10,
        ))

    # iNaturalist — terrestrial + freshwater species (all taxa)
    if inat and (scientificname or (lat is not None and lon is not None)):
        tasks["inat"] = inat.search(SearchParams(
            taxon=scientificname, lat=lat, lng=lon, radius_km=radius_km,
            start_date=start_date, end_date=end_date, limit=limit,
        ))

    # eBird — bird observations (if API key is configured)
    if ebird and getattr(ebird, '_api_key', None) and (lat is not None and lon is not None):
        tasks["ebird"] = ebird.search(SearchParams(
            taxon=scientificname, lat=lat, lng=lon, radius_km=radius_km,
            limit=limit,
        ))

    if include_climate and lat is not None and lon is not None and start_date and end_date:
        tasks["era5"] = era5.get_daily(
            lat=lat, lng=lon, start_date=start_date, end_date=end_date,
        )

    if not tasks:
        return {
            "error": "Please provide at least a species name or lat/lon coordinates.",
            "species_occurrences": [],
            "neon_sites": [],
            "climate": None,
        }

    results = {}
    task_keys = list(tasks.keys())
    task_coros = list(tasks.values())
    settled = await asyncio.gather(*task_coros, return_exceptions=True)

    for key, result in zip(task_keys, settled):
        if isinstance(result, Exception):
            logger.error("ecology_search: source %s failed: %s", key, result)
            results[key] = {"error": str(result)}
        else:
            results[key] = result

    scoring_params = SearchParams(
        taxon=scientificname, lat=lat, lng=lon, radius_km=radius_km,
        start_date=start_date, end_date=end_date, limit=limit,
    )

    # Format OBIS results with relevance scores
    obis_occurrences = []
    if "obis" in results and isinstance(results["obis"], list):
        for obs in results["obis"]:
            relevance = score_observation(obs, scoring_params)
            obis_occurrences.append({
                "id": obs.id,
                "scientific_name": obs.taxon.scientific_name if obs.taxon else None,
                "common_name": obs.taxon.common_name if obs.taxon else None,
                "lat": obs.location.lat,
                "lng": obs.location.lng,
                "observed_at": obs.observed_at.isoformat(),
                "depth_m": obs.value.get("depth_m") if obs.value else None,
                "basis_of_record": obs.value.get("basis_of_record") if obs.value else None,
                "quality_tier": obs.quality.tier,
                "license": obs.provenance.license,
                "source_url": obs.provenance.original_url,
                "relevance": {
                    "score": relevance.score,
                    "geo_distance_km": relevance.geo_distance_km,
                    "taxon_match": relevance.taxon_match,
                    "quality_score": relevance.quality_score,
                    "explanation": relevance.explanation,
                },
            })
        obis_occurrences.sort(key=lambda x: x["relevance"]["score"], reverse=True)

    # Format iNaturalist results with relevance scores
    inat_occurrences = []
    for source_key in ("inat", "ebird"):
        if source_key in results and isinstance(results[source_key], list):
            for obs in results[source_key]:
                relevance = score_observation(obs, scoring_params)
                inat_occurrences.append({
                    "id": obs.id,
                    "source": source_key,
                    "scientific_name": obs.taxon.scientific_name if obs.taxon else None,
                    "common_name": obs.taxon.common_name if obs.taxon else None,
                    "lat": obs.location.lat,
                    "lng": obs.location.lng,
                    "observed_at": obs.observed_at.isoformat(),
                    "quality_tier": obs.quality.tier,
                    "quality_grade": obs.quality.grade,
                    "media_url": obs.media_url,
                    "source_url": obs.provenance.original_url,
                    "relevance": {
                        "score": relevance.score,
                        "geo_distance_km": relevance.geo_distance_km,
                        "explanation": relevance.explanation,
                    },
                })
    inat_occurrences.sort(key=lambda x: x["relevance"]["score"], reverse=True)

    # Format NEON results
    neon_sites = []
    if "neon" in results and isinstance(results["neon"], list):
        for obs in results["neon"]:
            neon_sites.append({
                "site_code": obs.location.site_id,
                "site_name": obs.location.site_name,
                "lat": obs.location.lat,
                "lng": obs.location.lng,
                "state": obs.location.state_province,
                "data_products": obs.value.get("data_products_available") if obs.value else None,
            })

    # Format climate
    climate = None
    if "era5" in results and isinstance(results["era5"], dict):
        era5_raw = results["era5"]
        climate = {
            "source": "ERA5 (ECMWF)",
            "location_resolved": {
                "lat": era5_raw.get("latitude"),
                "lon": era5_raw.get("longitude"),
            },
            "daily": era5_raw.get("daily", {}),
            "units": era5_raw.get("daily_units", {}),
        }

    # Build search context
    search_context = {
        "query": {
            "scientificname": scientificname,
            "lat": lat,
            "lon": lon,
            "radius_km": radius_km,
            "date_range": f"{start_date} to {end_date}" if start_date and end_date else None,
        },
        "sources_queried": task_keys,
        "obis_records_returned": len(obis_occurrences),
        "neon_sites_found": len(neon_sites),
        "climate_included": climate is not None,
    }

    if not obis_occurrences and not neon_sites:
        logger.warning(
            "ecology_search: empty results for query taxon=%s, lat=%s, lon=%s",
            scientificname, lat, lon,
        )

    if obis_occurrences and len(obis_occurrences) < 3 and (lat is not None and lon is not None):
        search_context["sparse_results_hint"] = (
            f"Only {len(obis_occurrences)} records found. To get more: "
            f"expand radius_km (currently {radius_km or 'not set'}), "
            f"widen date range, or search at genus/family level."
        )

    # Merge all species occurrences (OBIS marine + iNat/eBird terrestrial)
    all_occurrences = obis_occurrences + inat_occurrences
    all_occurrences.sort(key=lambda x: x["relevance"]["score"], reverse=True)

    return {
        "species_occurrences": all_occurrences,
        "species_count": len(all_occurrences),
        "neon_sites": neon_sites,
        "neon_site_count": len(neon_sites),
        "climate": climate,
        "search_context": search_context,
    }


async def run_describe_sources(*, neon, obis, era5, inat=None, ebird=None) -> dict:
    """Describe all available ecological data sources and their capabilities."""
    sources = [neon, obis, era5]
    if inat:
        sources.append(inat)
    if ebird and getattr(ebird, '_api_key', None):
        sources.append(ebird)
    descriptions = []

    for adapter in sources:
        caps = adapter.capabilities()
        descriptions.append({
            "id": caps.adapter_id,
            "name": caps.name,
            "description": caps.description,
            "modalities": caps.modalities,
            "geographic_coverage": caps.geographic_coverage,
            "temporal_coverage_start": caps.temporal_coverage_start,
            "update_frequency": caps.update_frequency,
            "quality_tier": caps.quality_tier,
            "requires_auth": caps.requires_auth,
            "license": caps.license,
            "homepage": caps.homepage_url,
            "search_capabilities": {
                "location": caps.supports_location_search,
                "taxon": caps.supports_taxon_search,
                "date_range": caps.supports_date_range,
                "site_code": caps.supports_site_search,
            },
        })

    return {
        "source_count": len(descriptions),
        "sources": descriptions,
        "cross_source_tools": [
            {
                "name": "ecology_get_environmental_context",
                "description": "Get climate + sensors for a point and time (ERA5 + NEON combined)",
            },
            {
                "name": "ecology_search",
                "description": "Unified search across all sources — species, sites, and climate",
            },
        ],
    }
