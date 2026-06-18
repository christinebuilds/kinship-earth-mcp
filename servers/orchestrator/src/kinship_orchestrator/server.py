"""
Kinship Earth Orchestrator — cross-source ecological intelligence tools.

This is the unique value of Kinship Earth. No individual data source can
answer cross-domain questions like "what was the climate when this species
was observed?" or "what ecological data exists for this location?"

The orchestrator coordinates across NEON, OBIS, and ERA5 (and future sources)
to answer higher-level ecological questions.

Tools are unprefixed with 'ecology_' per the architecture conventions:
- ecology_get_environmental_context — climate + sensors for a point/time
- ecology_search — unified search across all sources
- ecology_describe_sources — what data is available

Run locally:   uv run mcp dev src/kinship_orchestrator/server.py
Run via HTTP:  uv run python -m kinship_orchestrator.server
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Literal, Optional

from mcp.server.fastmcp import FastMCP

from neonscience_mcp.adapter import NeonAdapter
from obis_mcp.adapter import OBISAdapter
from era5_mcp.adapter import ERA5Adapter
from inaturalist_mcp.adapter import INaturalistAdapter
from ebird_mcp.adapter import EBirdAdapter
from gbif_mcp.adapter import GBIFAdapter
from usgs_nwis_mcp.adapter import USGSNWISAdapter
from xenocanto_mcp.adapter import XenoCantoAdapter
from soilgrids_mcp.adapter import SoilGridsAdapter
from movebank_mcp.adapter import MovebankAdapter
from fluxnet_mcp.adapter import FLUXNETAdapter

from .auth_sqlite import SQLiteAuthManager

from kinship_shared import (
    ConversationTurn,
    EcologicalGraph,
    SearchParams,
    SQLiteConversationStore,
    run_describe_sources,
    run_get_environmental_context,
    run_search,
)
from kinship_shared.graph_extract import EntityExtractor

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "kinship-earth",
    instructions=(
        "This is the Kinship Earth orchestrator — the cross-source ecological intelligence "
        "layer. Use ecology_get_environmental_context to get climate, soil, and sensor data "
        "for any location and time period (combines ERA5 climate + nearest NEON sites). "
        "Use ecology_search to search for species observations AND their environmental "
        "context in a single call (combines OBIS marine data + NEON terrestrial data + ERA5 "
        "climate). Use ecology_describe_sources to learn what data sources are available "
        "and their capabilities. These tools answer questions that no single data source "
        "can answer alone."
    ),
)

# Initialize all adapters
_neon = NeonAdapter(api_token=os.environ.get("NEON_API_TOKEN"))
_obis = OBISAdapter()
_era5 = ERA5Adapter()
_inat = INaturalistAdapter()
_ebird = EBirdAdapter(api_key=os.environ.get("EBIRD_API_KEY"))
_gbif = GBIFAdapter()
_nwis = USGSNWISAdapter()
_xc = XenoCantoAdapter(api_key=os.environ.get("XC_API_KEY"))
_soil = SoilGridsAdapter()
_movebank = MovebankAdapter()
_fluxnet = FLUXNETAdapter()

# Conversation storage (fire-and-forget, never blocks tools)
_store = SQLiteConversationStore()
_conversation_id = os.environ.get("KINSHIP_CONVERSATION_ID", "")

# Auth (disabled by default, opt-in via KINSHIP_AUTH_ENABLED=true)
_auth = SQLiteAuthManager()
_auth_enabled = os.environ.get("KINSHIP_AUTH_ENABLED", "false").lower() == "true"
_auth_initialized = False

# Knowledge graph
_graph = EcologicalGraph()
_extractor = EntityExtractor()
_graph_initialized = False

# Lazy-initialize storage on first use
_store_initialized = False


async def _ensure_store() -> None:
    global _store_initialized
    if not _store_initialized:
        try:
            await _store.initialize()
            _store_initialized = True
        except Exception as e:
            logger.warning("Failed to initialize conversation store: %s", e)


async def _ensure_graph() -> None:
    global _graph_initialized
    if not _graph_initialized:
        try:
            await _graph.initialize()
            _graph_initialized = True
        except Exception as e:
            logger.warning("Failed to initialize graph: %s", e)


async def _store_turn(tool_name: str, params: dict, result: dict) -> None:
    """Persist a tool invocation and extract to graph. Fire-and-forget."""
    try:
        await _ensure_store()
        if not _store_initialized:
            return

        import uuid

        # Extract location from params
        lat = params.get("lat")
        lon = params.get("lon") or params.get("lng")

        # Extract taxa from params and results
        taxa = []
        if params.get("scientificname"):
            taxa.append(params["scientificname"])
        if params.get("scientific_name"):
            taxa.append(params["scientific_name"])
        if isinstance(result, dict):
            for occ in result.get("species_occurrences", [])[:5]:
                name = occ.get("scientific_name")
                if name and name not in taxa:
                    taxa.append(name)

        # Condense result to summary
        summary = {}
        if isinstance(result, dict):
            summary["species_count"] = result.get("species_count", 0)
            summary["neon_site_count"] = result.get("neon_site_count", 0)
            summary["climate_included"] = result.get("climate") is not None
            sources = result.get("search_context", {}).get("sources_queried", [])
            summary["sources_queried"] = sources

        turn = ConversationTurn(
            id=str(uuid.uuid4()),
            conversation_id=_conversation_id or str(uuid.uuid4()),
            user_id=os.environ.get("KINSHIP_USER_ID"),
            tool_name=tool_name,
            tool_params=params,
            tool_result_summary=summary,
            lat=lat,
            lng=lon,
            taxa_mentioned=taxa,
        )
        await _store.store_turn(turn)

        # Extract to knowledge graph
        if isinstance(result, dict):
            await _ensure_graph()
            if _graph_initialized:
                extracted = _extractor.extract_from_turn(turn, result)
                for entity in extracted.entities:
                    await _graph.add_entity(entity)
                for rel in extracted.relationships:
                    await _graph.add_relationship(rel)
                for fact in extracted.facts:
                    await _graph.add_fact(fact)
                await _graph.save()

    except Exception as e:
        logger.warning("Failed to store turn for %s: %s", tool_name, e)


import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def ecology_get_environmental_context(
    lat: float,
    lon: float,
    date: str,
    days_before: int = 7,
    days_after: int = 0,
) -> dict:
    """
    Get the full environmental context for a location and time.

    This is the flagship cross-source tool. Given a point on Earth and a date,
    it pulls climate data from ERA5 and finds the nearest NEON monitoring sites.
    Use this to understand the environmental conditions when and where a species
    was observed, or to characterize a study site.

    Args:
        lat: Latitude in decimal degrees (e.g. 45.82 for Wind River).
        lon: Longitude in decimal degrees (e.g. -121.95). Negative = West.
        date: The focal date in ISO 8601 format (e.g. '2023-06-15').
        days_before: Number of days before the focal date to include in the
                     climate window (default 7). Useful for seeing conditions
                     leading up to an observation.
        days_after: Number of days after the focal date to include (default 0).

    Returns:
        - ERA5 daily climate summary for the time window
        - Nearest NEON field sites (within 200km)
        - Location metadata
    """
    result = await run_get_environmental_context(
        lat=lat, lon=lon, date=date,
        days_before=days_before, days_after=days_after,
        neon=_neon, era5=_era5,
    )
    asyncio.create_task(_store_turn(
        "ecology_get_environmental_context",
        {"lat": lat, "lon": lon, "date": date, "days_before": days_before, "days_after": days_after},
        result,
    ))
    return result


@mcp.tool()
async def ecology_search(
    scientificname: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_climate: bool = True,
    limit: int = 20,
    offset: int = 0,
    output_format: Literal["json", "geojson"] = "json",
) -> dict:
    """
    Unified ecological search across all Kinship Earth data sources.

    Searches OBIS (marine species) and NEON (terrestrial sites) simultaneously,
    and optionally adds ERA5 climate context for the search area and time period.
    This is the single entry point for cross-source ecological data discovery.

    For source-specific queries with more control, use the individual servers:
    obis_search_occurrences (marine), inaturalist_search (all taxa),
    ebird_recent_observations (birds, real-time).

    Args:
        scientificname: Scientific name to search for (e.g. 'Delphinus delphis').
                        Searches OBIS occurrence records.
        lat: Latitude for geographic search (decimal degrees).
        lon: Longitude for geographic search (negative = West).
        radius_km: Search radius in kilometres.
        start_date: Start of date range (ISO 8601, e.g. '2020-01-01').
        end_date: End of date range (ISO 8601, e.g. '2023-12-31').
        include_climate: If true and lat/lon/dates are provided, include ERA5
                         daily climate summary for the search area (default true).
        limit: Max records per source (default 20).
        offset: Number of results to skip for pagination (default 0).
        output_format: "json" (default) returns structured dict with sections.
                       "geojson" returns species_occurrences as a GeoJSON
                       FeatureCollection (useful for QGIS and mapping tools).

    Example return (json): {"species_occurrences": [...], "neon_sites": [...],
    "climate": {"daily": [...]}, "search_context": {"relevance_scores": [...]}}
    """
    result = await run_search(
        scientificname=scientificname, lat=lat, lon=lon,
        radius_km=radius_km, start_date=start_date, end_date=end_date,
        include_climate=include_climate, limit=limit,
        obis=_obis, neon=_neon, era5=_era5,
        inat=_inat, ebird=_ebird,
    )

    if output_format == "geojson":
        from kinship_shared import observations_to_geojson
        occurrences = result.get("species_occurrences", [])
        result["species_occurrences_geojson"] = observations_to_geojson(occurrences)

    asyncio.create_task(_store_turn(
        "ecology_search",
        {"scientificname": scientificname, "lat": lat, "lon": lon, "radius_km": radius_km,
         "start_date": start_date, "end_date": end_date, "limit": limit},
        result,
    ))
    return result


@mcp.tool()
async def ecology_describe_sources() -> dict:
    """
    Describe all available ecological data sources and their capabilities.

    Returns a summary of each data source including what types of data it
    provides, geographic and temporal coverage, quality tier, and access
    requirements. Use this to understand what data is available before
    searching.
    """
    return await run_describe_sources(
        neon=_neon, obis=_obis, era5=_era5,
        inat=_inat, ebird=_ebird,
        gbif=_gbif, nwis=_nwis, xc=_xc,
        movebank=_movebank, fluxnet=_fluxnet,
    )


@mcp.tool()
async def ecology_whats_around_me(
    lat: float,
    lon: float,
    radius_km: float = 25,
    days_back: int = 7,
) -> dict:
    """
    Discover what's been observed near a location recently.

    A citizen-friendly entry point: "What's happening in the ecosystem around me?"
    Returns a snapshot combining recent species sightings, nearby monitoring sites,
    and current climate conditions from all available data sources.

    Args:
        lat: Your latitude in decimal degrees.
        lon: Your longitude in decimal degrees (negative = West).
        radius_km: How far to look (default 25 km).
        days_back: How many days of recent observations to include (default 7).

    Returns a combined ecological snapshot: recent species sightings (from eBird,
    iNaturalist, and OBIS), nearby monitoring sites (NEON), and current climate
    conditions (ERA5).
    """
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    result = await run_search(
        lat=lat, lon=lon,
        radius_km=radius_km,
        start_date=start_date,
        end_date=end_date,
        include_climate=True,
        limit=30,
        obis=_obis, neon=_neon, era5=_era5,
        inat=_inat, ebird=_ebird,
    )

    # Summarize by source for the citizen
    by_source = {}
    for occ in result.get("species_occurrences", []):
        src = occ.get("source", "obis")
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(occ)

    # Count unique species
    species_names = set()
    for occ in result.get("species_occurrences", []):
        name = occ.get("scientific_name")
        if name:
            species_names.add(name)

    return {
        "snapshot": {
            "location": {"lat": lat, "lon": lon, "radius_km": radius_km},
            "period": f"Last {days_back} days ({start_date} to {end_date})",
            "unique_species": len(species_names),
            "total_observations": len(result.get("species_occurrences", [])),
            "monitoring_sites_nearby": result.get("neon_site_count", 0),
            "climate_data_available": result.get("climate") is not None,
        },
        "recent_sightings": result.get("species_occurrences", [])[:20],
        "neon_sites": result.get("neon_sites", []),
        "climate": result.get("climate"),
        "sources_queried": result.get("search_context", {}).get("sources_queried", []),
    }


@mcp.tool()
async def ecology_graph_stats() -> dict:
    """
    Show knowledge graph statistics.

    Returns entity counts, relationship counts, top entities by mention
    count, and temporal fact counts. Useful for understanding how the
    ecological knowledge base is growing over time.
    """
    await _ensure_graph()
    if not _graph_initialized:
        return {"error": "Knowledge graph not available"}

    # Count by type
    entity_counts: dict[str, int] = {}
    top_entities: list[dict] = []

    for entity in _graph._entities.values():
        t = entity.entity_type
        entity_counts[t] = entity_counts.get(t, 0) + 1

    # Top 10 by mention count
    sorted_entities = sorted(
        _graph._entities.values(), key=lambda e: e.mention_count, reverse=True
    )
    for entity in sorted_entities[:10]:
        top_entities.append({
            "id": entity.id,
            "name": entity.name,
            "type": entity.entity_type,
            "mentions": entity.mention_count,
        })

    # Relationship counts by type
    rel_counts: dict[str, int] = {}
    if _graph._graph is not None:
        for _, _, data in _graph._graph.edges(data=True):
            rt = data.get("relationship_type", "unknown")
            rel_counts[rt] = rel_counts.get(rt, 0) + 1

    # Fact counts
    current_facts = sum(1 for f in _graph._facts.values() if f.valid_until is None)
    superseded_facts = sum(1 for f in _graph._facts.values() if f.valid_until is not None)

    return {
        "entities": {
            "total": _graph.entity_count(),
            "by_type": entity_counts,
        },
        "relationships": {
            "total": _graph.relationship_count(),
            "by_type": rel_counts,
        },
        "facts": {
            "total": _graph.fact_count(),
            "current": current_facts,
            "superseded": superseded_facts,
        },
        "top_entities": top_entities,
    }


# ---------------------------------------------------------------------------
# Ecosystem monitoring tools
# ---------------------------------------------------------------------------

from kinship_shared.anomaly_detect import run_anomaly_detection
from kinship_shared.anomaly_graph import anomaly_to_graph_entities
from kinship_shared.baselines import compute_baselines_from_era5, compute_baselines_from_usgs
from kinship_shared.monitoring import MonitoringRegistry, MonitoringSite
from kinship_shared.state_builder import build_ecosystem_state

_monitoring = MonitoringRegistry()
_monitoring_initialized = False


async def _ensure_monitoring() -> None:
    global _monitoring_initialized
    if not _monitoring_initialized:
        try:
            await _monitoring.initialize()
            _monitoring_initialized = True
        except Exception as e:
            logger.warning("Failed to initialize monitoring: %s", e)


@mcp.tool()
async def ecology_ecosystem_state(
    lat: float,
    lon: float,
    site_name: Optional[str] = None,
    period_days: int = 30,
) -> dict:
    """
    Get the current ecosystem state for a location.

    Computes a multi-signal state vector combining climate (ERA5),
    hydrology (USGS), and biodiversity data. Returns health score,
    deviation from baseline, and trend direction.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        site_name: Optional name to register this as a monitored site.
        period_days: Lookback window in days (default 30).
    """
    from kinship_shared.schema import Location

    await _ensure_monitoring()

    site_id = f"location:{lat:.2f}_{lon:.2f}"
    location = Location(lat=lat, lng=lon)

    # Get historical health scores for trend if site is monitored
    recent_scores: list[float] = []
    if _monitoring_initialized:
        history = await _monitoring.get_state_history(site_id, limit=10)
        recent_scores = [s.overall_health_score for s in history if s.overall_health_score is not None]

    state = await build_ecosystem_state(
        site_id=site_id,
        location=location,
        era5_adapter=_era5,
        usgs_adapter=_nwis,
        ebird_adapter=_ebird if getattr(_ebird, '_api_key', None) else None,
        gbif_adapter=_gbif,
        period_days=period_days,
        recent_health_scores=recent_scores,
    )

    # Store the state
    if _monitoring_initialized:
        await _monitoring.store_state(state)

    # Auto-register as monitoring site if named
    if site_name and _monitoring_initialized:
        site = await _monitoring.get_site(site_id)
        if not site:
            await _monitoring.add_site(MonitoringSite(
                site_id=site_id, name=site_name, location=location,
            ))

    result = state.model_dump(mode="json")
    result["visualization"] = {"primary": "dashboard", "description": f"Ecosystem state for ({lat}, {lon})"}

    asyncio.create_task(_store_turn(
        "ecology_ecosystem_state",
        {"lat": lat, "lon": lon, "period_days": period_days},
        result,
    ))
    return result


@mcp.tool()
async def ecology_monitor_site(
    action: Literal["add", "remove", "list"] = "list",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    site_name: Optional[str] = None,
    site_id: Optional[str] = None,
) -> dict:
    """
    Manage monitored ecosystem sites.

    Args:
        action: 'add' to register, 'remove' to stop monitoring, 'list' to show all.
        lat: Latitude (required for 'add').
        lon: Longitude (required for 'add').
        site_name: Human-readable name (required for 'add').
        site_id: Site ID (required for 'remove', e.g. 'location:41.50_-70.70').
    """
    from kinship_shared.schema import Location

    await _ensure_monitoring()
    if not _monitoring_initialized:
        return {"error": "Monitoring registry not available"}

    if action == "add":
        if lat is None or lon is None or not site_name:
            return {"error": "lat, lon, and site_name are required for 'add'"}
        sid = site_id or f"location:{lat:.2f}_{lon:.2f}"
        site = MonitoringSite(
            site_id=sid, name=site_name,
            location=Location(lat=lat, lng=lon),
        )
        await _monitoring.add_site(site)
        return {"status": "ok", "action": "added", "site_id": sid, "name": site_name}

    elif action == "remove":
        if not site_id:
            return {"error": "site_id is required for 'remove'"}
        removed = await _monitoring.remove_site(site_id)
        return {"status": "ok" if removed else "not_found", "action": "removed", "site_id": site_id}

    else:  # list
        sites = await _monitoring.list_sites()
        site_list = []
        for s in sites:
            latest = await _monitoring.get_latest_state(s.site_id)
            site_list.append({
                "site_id": s.site_id,
                "name": s.name,
                "lat": s.location.lat,
                "lng": s.location.lng,
                "last_checked": s.last_checked.isoformat() if s.last_checked else None,
                "health_score": latest.overall_health_score if latest else None,
                "trend": latest.trend_direction if latest else None,
            })
        return {"sites": site_list, "count": len(site_list)}


@mcp.tool()
async def ecology_check_anomalies(
    lat: float,
    lon: float,
    severity_min: Literal["info", "warning", "critical"] = "info",
) -> dict:
    """
    Check for ecological anomalies at a location.

    Runs the anomaly detection pipeline: fetches current ecosystem state,
    computes baselines, and identifies deviations in temperature, streamflow,
    species composition, and phenology.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        severity_min: Minimum severity to return ('info', 'warning', 'critical').
    """
    from kinship_shared.schema import Location

    location = Location(lat=lat, lng=lon)
    site_id = f"location:{lat:.2f}_{lon:.2f}"

    # Build ecosystem state
    state = await build_ecosystem_state(
        site_id=site_id, location=location,
        era5_adapter=_era5, usgs_adapter=_nwis,
        ebird_adapter=_ebird if getattr(_ebird, '_api_key', None) else None,
        gbif_adapter=_gbif,
    )

    # Compute baselines
    from datetime import datetime as dt, timezone as tz
    now = dt.now(tz.utc)
    baseline = await compute_baselines_from_era5(_era5, lat, lon, now)

    # Run detection
    anomalies = run_anomaly_detection(
        location=location, baseline=baseline, state=state,
    )

    # Filter by severity
    severity_order = {"info": 0, "warning": 1, "critical": 2}
    min_level = severity_order.get(severity_min, 0)
    filtered = [a for a in anomalies if severity_order.get(a.severity, 0) >= min_level]

    # Wire into graph
    await _ensure_graph()
    if _graph_initialized:
        for anomaly in filtered:
            entities, rels = anomaly_to_graph_entities(anomaly)
            for e in entities:
                await _graph.add_entity(e)
            for r in rels:
                await _graph.add_relationship(r)
        if filtered:
            await _graph.save()

    result = {
        "location": {"lat": lat, "lon": lon},
        "anomaly_count": len(filtered),
        "anomalies": [a.model_dump(mode="json") for a in filtered],
        "ecosystem_state_summary": {
            "health_score": state.overall_health_score,
            "trend": state.trend_direction,
            "sources": state.sources_contributing,
        },
        "visualization": {"primary": "dashboard", "description": f"Anomaly check for ({lat}, {lon})"},
    }

    asyncio.create_task(_store_turn("ecology_check_anomalies", {"lat": lat, "lon": lon}, result))
    return result


@mcp.tool()
async def ecology_events(
    lat: float,
    lng: float,
    include_analogs: bool = True,
    severity_min: Literal["info", "warning", "critical"] = "info",
) -> dict:
    """
    Synthesize ecological events from detected anomalies at a location.

    Runs the full pipeline: ecosystem state → anomaly detection → event
    synthesis → historical analog matching. Returns classified events like
    'drought cascade', 'die-off', 'phenological shift' with data-grounded
    narratives and historical context.

    This is the highest-level ecological intelligence tool — it interprets
    raw signals into ecological stories.

    Args:
        lat: Latitude in decimal degrees.
        lng: Longitude in decimal degrees.
        include_analogs: Whether to search for historical analogs (default True).
        severity_min: Minimum event severity to return ('info', 'warning', 'critical').
    """
    from kinship_shared.schema import Location
    from kinship_shared.event_classify import synthesize_events
    from kinship_shared.analog_match import find_historical_analogs, attach_analog_to_event

    location = Location(lat=lat, lng=lng)
    site_id = f"location:{lat:.2f}_{lng:.2f}"

    state = await build_ecosystem_state(
        site_id=site_id, location=location,
        era5_adapter=_era5, usgs_adapter=_nwis,
        ebird_adapter=_ebird if getattr(_ebird, '_api_key', None) else None,
        gbif_adapter=_gbif,
    )

    from datetime import datetime as dt, timezone as tz
    now = dt.now(tz.utc)
    baseline = await compute_baselines_from_era5(_era5, lat, lng, now)

    anomalies = run_anomaly_detection(
        location=location, baseline=baseline, state=state,
    )

    severity_order = {"info": 0, "warning": 1, "critical": 2}
    min_level = severity_order.get(severity_min, 0)
    filtered_anomalies = [a for a in anomalies if severity_order.get(a.severity, 0) >= min_level]

    events = synthesize_events(filtered_anomalies, location=location)

    all_analogs: list[dict] = []
    if include_analogs:
        for event in events:
            analogs = find_historical_analogs(event)
            attach_analog_to_event(event, analogs)
            all_analogs.extend([a.model_dump(mode="json") for a in analogs])

    result = {
        "location": {"lat": lat, "lng": lng},
        "event_count": len(events),
        "events": [e.model_dump(mode="json") for e in events],
        "anomaly_count": len(filtered_anomalies),
        "historical_analogs": all_analogs,
        "visualization_hint": "timeline",
    }

    asyncio.create_task(_store_turn("ecology_events", {"lat": lat, "lng": lng}, result))
    return result


@mcp.tool()
async def ecology_subscribe(
    action: Literal["subscribe", "unsubscribe", "list"],
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    site_name: Optional[str] = None,
    severity_min: str = "warning",
) -> dict:
    """
    Subscribe to ecological alerts for a location.

    - subscribe: Register for alerts when anomalies/events are detected
    - unsubscribe: Stop receiving alerts for a location
    - list: Show all active subscriptions

    Subscriptions are stored in the monitoring registry. When the
    ecology_check_anomalies or ecology_events tools detect signals above
    the severity threshold, subscribed locations are flagged.

    Args:
        action: One of 'subscribe', 'unsubscribe', 'list'.
        lat: Latitude (required for subscribe/unsubscribe).
        lng: Longitude (required for subscribe/unsubscribe).
        site_name: Optional human-readable name for the location.
        severity_min: Minimum severity to alert on ('warning' default).
    """
    from kinship_shared.monitoring import Subscription
    from kinship_shared.schema import Location

    await _ensure_monitoring()

    if action == "list":
        subs = await _monitoring.list_subscriptions()
        pending = await _monitoring.get_pending_alerts()
        return {
            "action": "list",
            "subscriptions": [s.model_dump(mode="json") for s in subs],
            "pending_alerts": pending,
            "count": len(subs),
        }

    if lat is None or lng is None:
        return {"error": "lat and lng are required for subscribe/unsubscribe"}

    site_id = f"location:{lat:.2f}_{lng:.2f}"

    if action == "subscribe":
        site = MonitoringSite(
            site_id=site_id,
            name=site_name or f"{lat:.2f}, {lng:.2f}",
            location=Location(lat=lat, lng=lng),
        )
        await _monitoring.add_site(site)
        sub = Subscription(site_id=site_id, severity_min=severity_min)
        await _monitoring.add_subscription(sub)
        return {
            "action": "subscribe",
            "site_id": site_id,
            "site_name": site.name,
            "severity_min": severity_min,
            "message": f"Subscribed to alerts for {site.name} (severity >= {severity_min})",
        }

    if action == "unsubscribe":
        removed = await _monitoring.remove_subscription(site_id, "anonymous")
        return {
            "action": "unsubscribe",
            "site_id": site_id,
            "removed": removed,
            "message": f"Unsubscribed from alerts for {site_id}" if removed else f"No subscription found for {site_id}",
        }

    return {"error": f"Unknown action: {action}"}


@mcp.tool()
async def ecology_memory_recall(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: float = 50,
    scientific_name: Optional[str] = None,
    depth: int = 2,
) -> dict:
    """
    Recall what the knowledge graph knows about a location or species.

    Searches the ecological memory for entities, relationships, and
    temporal facts. Returns connected entities, co-occurring species,
    and historical facts. The graph grows with every query — the more
    the system is used, the richer the recall.

    Args:
        lat: Latitude to recall memory for.
        lon: Longitude to recall memory for.
        radius_km: Search radius (default 50 km).
        scientific_name: Species to recall memory for.
        depth: How many relationship hops to traverse (default 2).
    """
    from kinship_shared.graph_schema import make_species_id, make_location_id

    await _ensure_graph()
    if not _graph_initialized:
        return {"memory": [], "message": "Knowledge graph not available"}

    results: dict = {"query": {"lat": lat, "lon": lon, "scientific_name": scientific_name}}

    # Species recall
    if scientific_name:
        species_id = make_species_id(scientific_name)
        entity = await _graph.get_entity(species_id)
        if entity:
            neighbors = await _graph.get_neighbors(species_id, depth=depth)
            co_occurring = await _graph.find_co_occurring_species(species_id, min_evidence=1)
            facts = await _graph.get_current_facts(species_id)
            results["species"] = {
                "id": species_id,
                "name": entity.name,
                "mentions": entity.mention_count,
                "neighbors": neighbors.get("neighbors", []),
                "co_occurring_species": co_occurring,
                "facts": [f.model_dump(mode="json") for f in facts],
            }
        else:
            results["species"] = {"id": species_id, "found": False, "message": f"No memory of {scientific_name} yet"}

    # Location recall
    if lat is not None and lon is not None:
        location_id = make_location_id(lat, lon)
        entity = await _graph.get_entity(location_id)
        interest = await _graph.get_location_interest(lat, lon, radius_km)
        if entity:
            neighbors = await _graph.get_neighbors(location_id, depth=depth)
            facts = await _graph.get_current_facts(location_id)
            results["location"] = {
                "id": location_id,
                "mentions": entity.mention_count,
                "interest": interest,
                "neighbors": neighbors.get("neighbors", []),
                "facts": [f.model_dump(mode="json") for f in facts],
            }
        else:
            results["location"] = {"id": location_id, "found": False, "interest": interest}

    results["graph_size"] = {
        "entities": _graph.entity_count(),
        "relationships": _graph.relationship_count(),
        "facts": _graph.fact_count(),
    }

    return results


@mcp.tool()
async def ecology_memory_store(
    name: str,
    description: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    scientific_name: Optional[str] = None,
    share: bool = False,
) -> dict:
    """
    Explicitly save an insight to the ecological knowledge graph.

    Use this when you discover something worth remembering — a confirmed
    spawning site, an unusual species observation, a climate correlation.
    The insight becomes a graph entity that enriches future queries.

    Args:
        name: Short name (e.g. 'Coho spawning confirmed at Russian River tributary').
        description: Detailed description of the finding.
        lat: Latitude (if location-relevant).
        lon: Longitude (if location-relevant).
        scientific_name: Species name (if species-relevant).
        share: If True, this finding is visible to other users (opt-in shared memory).
    """
    from kinship_shared.graph_schema import GraphEntity, GraphRelationship, make_species_id, make_location_id
    import uuid

    await _ensure_graph()
    if not _graph_initialized:
        return {"error": "Knowledge graph not available"}

    finding_id = f"finding:{uuid.uuid4()}"
    user_id = os.environ.get("KINSHIP_USER_ID", "anonymous")

    finding = GraphEntity(
        id=finding_id,
        entity_type="finding",
        name=name,
        properties={
            "description": description,
            "user_id": user_id,
            "shared": share,
        },
    )
    await _graph.add_entity(finding)

    # Link to location
    if lat is not None and lon is not None:
        loc_id = make_location_id(lat, lon)
        loc_entity = GraphEntity(
            id=loc_id, entity_type="location",
            name=f"({lat:.2f}, {lon:.2f})",
            properties={"lat": lat, "lng": lon},
        )
        await _graph.add_entity(loc_entity)
        await _graph.add_relationship(GraphRelationship(
            source_id=finding_id, target_id=loc_id,
            relationship_type="FOUND_IN",
        ))

    # Link to species
    if scientific_name:
        sp_id = make_species_id(scientific_name)
        sp_entity = GraphEntity(
            id=sp_id, entity_type="species", name=scientific_name,
        )
        await _graph.add_entity(sp_entity)
        await _graph.add_relationship(GraphRelationship(
            source_id=finding_id, target_id=sp_id,
            relationship_type="QUERIED_ABOUT",
        ))

    await _graph.save()

    return {
        "status": "ok",
        "finding_id": finding_id,
        "name": name,
        "shared": share,
        "linked_to": {
            "location": make_location_id(lat, lon) if lat and lon else None,
            "species": make_species_id(scientific_name) if scientific_name else None,
        },
    }


@mcp.tool()
async def ecology_related_queries(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    scientific_name: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """
    Find related past queries from the knowledge graph.

    Shows what other queries have been made about a location or species.
    Surfaces patterns like "3 researchers queried this watershed last
    month" or "this species was also searched near 5 other locations."

    Args:
        lat: Latitude to find related queries for.
        lon: Longitude to find related queries for.
        scientific_name: Species to find related queries for.
        limit: Max related queries to return (default 10).
    """
    from kinship_shared.graph_schema import make_species_id, make_location_id

    await _ensure_graph()
    if not _graph_initialized:
        return {"related_queries": [], "message": "Knowledge graph not available"}

    target_ids = []
    if scientific_name:
        target_ids.append(make_species_id(scientific_name))
    if lat is not None and lon is not None:
        target_ids.append(make_location_id(lat, lon))

    related = []
    researchers = set()

    for target_id in target_ids:
        rels = await _graph.get_relationships(target_id, rel_type="QUERIED_ABOUT")
        for rel in rels[:limit]:
            query_entity = await _graph.get_entity(rel.source_id)
            if query_entity and query_entity.entity_type == "query":
                # Find who made the query
                query_rels = await _graph.get_relationships(rel.source_id, rel_type="QUERIED_BY")
                for qr in query_rels:
                    researchers.add(qr.target_id)

                related.append({
                    "query_id": rel.source_id,
                    "name": query_entity.name,
                    "timestamp": query_entity.properties.get("timestamp", ""),
                    "mentions": query_entity.mention_count,
                })

    return {
        "target": {"scientific_name": scientific_name, "lat": lat, "lon": lon},
        "related_queries": related[:limit],
        "unique_researchers": len(researchers),
        "total_related": len(related),
    }


@mcp.tool()
async def ecology_emerging_patterns(
    min_mentions: int = 3,
    limit: int = 10,
) -> dict:
    """
    Surface emerging patterns from the ecological knowledge graph.

    Finds entities with growing mention counts, species with expanding
    co-occurrence networks, and locations with increasing research
    interest. These are signals of ecological change or research momentum.

    Args:
        min_mentions: Minimum mention count to include (default 3).
        limit: Max patterns to return (default 10).
    """
    await _ensure_graph()
    if not _graph_initialized:
        return {"patterns": [], "message": "Knowledge graph not available"}

    # Top species by mentions
    top_species = []
    for entity in sorted(_graph._entities.values(), key=lambda e: e.mention_count, reverse=True):
        if entity.entity_type == "species" and entity.mention_count >= min_mentions:
            co = await _graph.find_co_occurring_species(entity.id, min_evidence=1)
            top_species.append({
                "id": entity.id,
                "name": entity.name,
                "mentions": entity.mention_count,
                "co_occurring_species_count": len(co),
            })
            if len(top_species) >= limit:
                break

    # Top locations by mentions
    top_locations = []
    for entity in sorted(_graph._entities.values(), key=lambda e: e.mention_count, reverse=True):
        if entity.entity_type == "location" and entity.mention_count >= min_mentions:
            top_locations.append({
                "id": entity.id,
                "name": entity.name,
                "mentions": entity.mention_count,
                "properties": entity.properties,
            })
            if len(top_locations) >= limit:
                break

    # Strongest relationships
    top_relationships = []
    if _graph._graph is not None:
        edges = []
        for s, t, data in _graph._graph.edges(data=True):
            edges.append((s, t, data.get("evidence_count", 1), data.get("relationship_type", "")))
        edges.sort(key=lambda x: x[2], reverse=True)
        for s, t, count, rtype in edges[:limit]:
            if count >= min_mentions:
                top_relationships.append({
                    "source": s, "target": t,
                    "type": rtype, "evidence_count": count,
                })

    return {
        "top_species": top_species,
        "top_locations": top_locations,
        "strongest_relationships": top_relationships,
        "graph_size": {
            "entities": _graph.entity_count(),
            "relationships": _graph.relationship_count(),
        },
    }


@mcp.tool()
async def ecology_biodiversity_assessment(
    lat: float,
    lon: float,
    radius_km: float = 25,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    Comprehensive biodiversity assessment for a location.

    Chains species search + climate + soil into a structured assessment.
    Returns species richness, taxonomic diversity, environmental context,
    and data quality metrics — everything needed for a baseline survey.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees (negative = West).
        radius_km: Search radius in km (default 25).
        start_date: Start of date range (ISO 8601). Defaults to 90 days ago.
        end_date: End of date range (ISO 8601). Defaults to today.
    """
    from datetime import datetime as dt, timedelta

    if not end_date:
        end_date = dt.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (dt.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    search_result, soil_result = await asyncio.gather(
        run_search(
            lat=lat, lon=lon, radius_km=radius_km,
            start_date=start_date, end_date=end_date,
            include_climate=True, limit=50,
            obis=_obis, neon=_neon, era5=_era5,
            inat=_inat, ebird=_ebird,
        ),
        _soil.search(SearchParams(lat=lat, lng=lon, radius_km=1, limit=1)),
    )

    occurrences = search_result.get("species_occurrences", [])

    # Compute derived metrics
    species_set = set()
    taxonomic_groups: dict[str, set] = {}
    quality_dist: dict[int, int] = {}
    observation_dates = []
    sources_seen: set[str] = set()

    for occ in occurrences:
        name = occ.get("scientific_name")
        if name:
            species_set.add(name)
        source = occ.get("source", "obis")
        sources_seen.add(source)
        tier = occ.get("quality_tier")
        if tier is not None:
            quality_dist[tier] = quality_dist.get(tier, 0) + 1
        obs_date = occ.get("observed_at", "")[:10]
        if obs_date:
            observation_dates.append(obs_date)

    soil_data = {}
    if soil_result:
        soil_data = soil_result[0].value if soil_result[0].value else {}

    assessment = {
        "assessment_type": "biodiversity_assessment",
        "location": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "period": {"start": start_date, "end": end_date},
        "metrics": {
            "species_richness": len(species_set),
            "total_observations": len(occurrences),
            "source_coverage": sorted(sources_seen),
            "quality_distribution": quality_dist,
            "temporal_coverage": {
                "earliest": min(observation_dates) if observation_dates else None,
                "latest": max(observation_dates) if observation_dates else None,
            },
        },
        "species_list": sorted(species_set),
        "observations": occurrences,
        "neon_sites": search_result.get("neon_sites", []),
        "climate": search_result.get("climate"),
        "soil": soil_data,
    }

    asyncio.create_task(_store_turn(
        "ecology_biodiversity_assessment",
        {"lat": lat, "lon": lon, "radius_km": radius_km, "start_date": start_date, "end_date": end_date},
        assessment,
    ))
    return assessment


@mcp.tool()
async def ecology_temporal_comparison(
    lat: float,
    lon: float,
    radius_km: float = 50,
    period_a_start: str = "",
    period_a_end: str = "",
    period_b_start: str = "",
    period_b_end: str = "",
    scientificname: Optional[str] = None,
) -> dict:
    """
    Compare ecological conditions between two time periods.

    Answers "what changed here?" by running parallel queries for both
    periods and computing differences in species composition, climate,
    and observation patterns.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        radius_km: Search radius (default 50 km).
        period_a_start: Start of period A (ISO 8601, e.g. '2015-01-01').
        period_a_end: End of period A (e.g. '2018-12-31').
        period_b_start: Start of period B (e.g. '2021-01-01').
        period_b_end: End of period B (e.g. '2024-12-31').
        scientificname: Optional species filter.
    """
    if not period_a_start or not period_a_end or not period_b_start or not period_b_end:
        return {"error": "All four date parameters are required (period_a_start, period_a_end, period_b_start, period_b_end)"}

    result_a, result_b = await asyncio.gather(
        run_search(
            scientificname=scientificname, lat=lat, lon=lon, radius_km=radius_km,
            start_date=period_a_start, end_date=period_a_end,
            include_climate=True, limit=50,
            obis=_obis, neon=_neon, era5=_era5,
            inat=_inat, ebird=_ebird,
        ),
        run_search(
            scientificname=scientificname, lat=lat, lon=lon, radius_km=radius_km,
            start_date=period_b_start, end_date=period_b_end,
            include_climate=True, limit=50,
            obis=_obis, neon=_neon, era5=_era5,
            inat=_inat, ebird=_ebird,
        ),
    )

    # Extract species sets
    species_a = {occ.get("scientific_name") for occ in result_a.get("species_occurrences", []) if occ.get("scientific_name")}
    species_b = {occ.get("scientific_name") for occ in result_b.get("species_occurrences", []) if occ.get("scientific_name")}

    # Climate deltas
    climate_delta = {}
    climate_a = result_a.get("climate", {})
    climate_b = result_b.get("climate", {})
    if climate_a and climate_b:
        daily_a = climate_a.get("daily", {})
        daily_b = climate_b.get("daily", {})
        for var in ["temperature_2m_mean", "precipitation_sum"]:
            vals_a = daily_a.get(var, [])
            vals_b = daily_b.get(var, [])
            if vals_a and vals_b:
                avg_a = sum(vals_a) / len(vals_a)
                avg_b = sum(vals_b) / len(vals_b)
                climate_delta[var] = {
                    "period_a_avg": round(avg_a, 2),
                    "period_b_avg": round(avg_b, 2),
                    "delta": round(avg_b - avg_a, 2),
                }

    comparison = {
        "comparison_type": "temporal_comparison",
        "location": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "species_filter": scientificname,
        "period_a": {"start": period_a_start, "end": period_a_end},
        "period_b": {"start": period_b_start, "end": period_b_end},
        "deltas": {
            "species_gained": sorted(species_b - species_a),
            "species_lost": sorted(species_a - species_b),
            "species_persistent": sorted(species_a & species_b),
            "observation_count_a": result_a.get("species_count", 0),
            "observation_count_b": result_b.get("species_count", 0),
            "observation_count_change": result_b.get("species_count", 0) - result_a.get("species_count", 0),
            "climate": climate_delta,
        },
        "period_a_data": result_a,
        "period_b_data": result_b,
    }

    asyncio.create_task(_store_turn(
        "ecology_temporal_comparison",
        {"lat": lat, "lon": lon, "period_a": f"{period_a_start}/{period_a_end}", "period_b": f"{period_b_start}/{period_b_end}"},
        comparison,
    ))
    return comparison


@mcp.tool()
async def ecology_export(
    format: Literal["csv", "geojson", "markdown", "bibtex"] = "geojson",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    scientificname: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    Export ecological data in a standard format.

    Runs a search and formats the output as CSV, GeoJSON, Markdown
    report, or BibTeX citations with proper provenance.

    Args:
        format: Output format — 'csv', 'geojson', 'markdown', or 'bibtex'.
        lat: Latitude for geographic search.
        lon: Longitude for geographic search.
        radius_km: Search radius in km.
        scientificname: Species to search for.
        start_date: Start of date range.
        end_date: End of date range.
    """
    from kinship_shared.export import to_csv, to_geojson, to_markdown, to_bibtex

    # Run search
    result = await run_search(
        scientificname=scientificname, lat=lat, lon=lon,
        radius_km=radius_km, start_date=start_date, end_date=end_date,
        include_climate=True, limit=50,
        obis=_obis, neon=_neon, era5=_era5,
        inat=_inat, ebird=_ebird,
    )

    observations = result.get("species_occurrences", [])
    params = {"scientificname": scientificname, "lat": lat, "lon": lon,
              "radius_km": radius_km, "start_date": start_date, "end_date": end_date}
    sources = result.get("search_context", {}).get("sources_queried", [])

    if format == "csv":
        return {"format": "csv", "content": to_csv(observations, params), "record_count": len(observations)}
    elif format == "geojson":
        return {"format": "geojson", "content": to_geojson(observations, params), "record_count": len(observations)}
    elif format == "markdown":
        return {"format": "markdown", "content": to_markdown(observations, result.get("climate"), sources, params), "record_count": len(observations)}
    elif format == "bibtex":
        return {"format": "bibtex", "content": to_bibtex(sources), "source_count": len(sources)}
    else:
        return {"error": f"Unknown format: {format}"}


@mcp.tool()
async def ecology_cite(
    sources: Optional[str] = None,
) -> dict:
    """
    Generate citations for ecological data sources.

    Returns properly formatted citations (BibTeX + APA) for OBIS, NEON,
    ERA5, and all other data sources. Includes DOIs, access dates, and
    license info.

    Args:
        sources: Comma-separated source IDs to cite (e.g. 'obis,era5,neon').
                 If omitted, returns citations for all available sources.
    """
    from kinship_shared.citations import get_citations

    source_list = None
    if sources:
        source_list = [s.strip() for s in sources.split(",")]

    return get_citations(source_list)


@mcp.tool()
async def ecology_my_history(
    limit: int = 20,
    taxon_filter: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
) -> dict:
    """
    View your past ecological queries.

    Returns your recent query history, optionally filtered by species
    or location. Useful for picking up where you left off or seeing
    what you've explored.

    Args:
        limit: Maximum number of turns to return (default 20).
        taxon_filter: Filter by scientific name (e.g. 'Delphinus delphis').
        lat: Filter by latitude (requires lon and radius_km).
        lon: Filter by longitude.
        radius_km: Search radius for location filter.
    """
    await _ensure_store()
    if not _store_initialized:
        return {"turns": [], "message": "Conversation storage not available"}

    if taxon_filter:
        turns = await _store.get_turns_by_taxon(taxon_filter, limit=limit)
    elif lat is not None and lon is not None and radius_km:
        turns = await _store.get_turns_by_location(lat, lon, radius_km, limit=limit)
    else:
        # Get recent turns for the current conversation
        conv_id = _conversation_id or ""
        if conv_id:
            turns = await _store.get_conversation(conv_id)
            turns = turns[-limit:]
        else:
            turns = []

    # Collect stats
    all_taxa = set()
    all_locations = set()
    for t in turns:
        all_taxa.update(t.taxa_mentioned)
        if t.lat is not None and t.lng is not None:
            all_locations.add((round(t.lat, 2), round(t.lng, 2)))

    return {
        "turns": [
            {
                "id": t.id,
                "timestamp": t.timestamp.isoformat(),
                "tool_name": t.tool_name,
                "summary": t.tool_result_summary,
                "location": {"lat": t.lat, "lon": t.lng} if t.lat else None,
                "taxa": t.taxa_mentioned,
                "feedback": t.feedback,
            }
            for t in turns
        ],
        "total_queries": len(turns),
        "unique_species_queried": len(all_taxa),
        "unique_locations_queried": len(all_locations),
    }


@mcp.tool()
async def ecology_my_usage() -> dict:
    """
    Check your usage and rate limit status.

    Returns your current query count, daily limit, tier, and
    stored API keys (names only, not values).
    """
    global _auth_initialized
    try:
        if not _auth_initialized:
            await _auth.initialize()
            _auth_initialized = True

        user_id = os.environ.get("KINSHIP_USER_ID", "default")
        user = await _auth.get_user(user_id)

        if user:
            return {
                "user": {
                    "id": user.id,
                    "email": user.email or "(anonymous)",
                    "tier": user.tier,
                    "queries_today": user.queries_today,
                    "queries_limit": user.queries_limit,
                    "api_keys_configured": list(user.api_keys.keys()),
                },
            }
        else:
            return {
                "user": {
                    "id": user_id,
                    "email": "(anonymous)",
                    "tier": "free",
                    "queries_today": 0,
                    "queries_limit": 50,
                    "api_keys_configured": [],
                },
                "message": "No user profile found. Queries are not being metered.",
            }
    except Exception as e:
        logger.warning("Failed to get usage: %s", e)
        return {"error": f"Failed to get usage: {e}"}


@mcp.tool()
async def ecology_set_api_key(
    service: Literal["ebird", "xeno-canto", "neon"],
    api_key: str,
) -> dict:
    """
    Store your own API key for a data source (BYOK).

    Some data sources (eBird, Xeno-canto) require API keys for full access.
    Provide your own key, which will be stored locally and used for your
    queries. Keys are stored in ~/.kinship-earth/users.db.

    Args:
        service: The data source to set the key for ('ebird', 'xeno-canto', or 'neon').
        api_key: Your API key for the service.
    """
    global _auth_initialized
    try:
        if not _auth_initialized:
            await _auth.initialize()
            _auth_initialized = True

        user_id = os.environ.get("KINSHIP_USER_ID", "default")
        user = await _auth.get_user(user_id)
        if user is None:
            user = await _auth.create_anonymous_user()

        await _auth.set_api_key(user.id, service, api_key)
        return {"status": "ok", "service": service, "message": f"API key stored for {service}"}
    except Exception as e:
        logger.warning("Failed to store API key: %s", e)
        return {"error": f"Failed to store API key: {e}"}


@mcp.tool()
async def ecology_feedback(
    turn_id: str,
    feedback: str,
) -> dict:
    """
    Provide feedback on a previous query result.

    Attaches feedback to a stored conversation turn. Use 'helpful',
    'not_helpful', or free-text feedback. This helps improve data
    quality and ranking over time.

    Args:
        turn_id: The ID of the conversation turn to provide feedback on.
        feedback: Feedback text — 'helpful', 'not_helpful', or free text.
    """
    await _ensure_store()
    if not _store_initialized:
        return {"error": "Conversation storage not available"}

    found = await _store.add_feedback(turn_id, feedback)
    if found:
        return {"status": "ok", "turn_id": turn_id, "feedback": feedback}
    else:
        return {"error": f"Turn {turn_id} not found"}


# ---------------------------------------------------------------------------
# Prompts — curated multi-step workflows for agents
# ---------------------------------------------------------------------------


@mcp.prompt()
async def ecological_survey(
    lat: float,
    lon: float,
    radius_km: float = 25,
    start_date: str = "",
    end_date: str = "",
) -> str:
    """
    Comprehensive biodiversity + climate + soil report for a location.

    Runs a multi-source survey: species observations (OBIS, iNaturalist,
    eBird), NEON monitoring sites, ERA5 climate, and SoilGrids soil
    properties. Returns a structured dataset the agent should synthesize
    into a narrative ecological survey report.
    """
    from datetime import datetime, timedelta

    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    search_result = await run_search(
        lat=lat, lon=lon, radius_km=radius_km,
        start_date=start_date, end_date=end_date,
        include_climate=True, limit=30,
        obis=_obis, neon=_neon, era5=_era5,
        inat=_inat, ebird=_ebird,
    )

    soil_result = await _soil.search(SearchParams(lat=lat, lng=lon, radius_km=1, limit=1))
    soil_data = {}
    if soil_result:
        obs = soil_result[0]
        soil_data = obs.value if obs.value else {}

    survey_data = {
        "survey_type": "ecological_survey",
        "location": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "period": {"start": start_date, "end": end_date},
        "species_observations": search_result.get("species_occurrences", []),
        "species_count": search_result.get("species_count", 0),
        "neon_sites": search_result.get("neon_sites", []),
        "climate": search_result.get("climate"),
        "soil": soil_data,
        "sources_queried": search_result.get("search_context", {}).get("sources_queried", []),
    }

    return (
        "You are conducting an ecological survey. Below is data from 9 ecological "
        "data sources for the requested location and time period. Synthesize this "
        "into a comprehensive ecological survey report with these sections:\n\n"
        "1. **Location Overview** — coordinates, nearest monitoring sites, ecosystem type\n"
        "2. **Biodiversity Summary** — species observed, taxonomic diversity, notable species\n"
        "3. **Climate Conditions** — temperature, precipitation, wind patterns for the period\n"
        "4. **Soil Properties** — composition, organic carbon, pH, moisture (if available)\n"
        "5. **Data Quality Assessment** — source mix, quality tiers, confidence levels\n"
        "6. **Gaps & Recommendations** — what data is missing, suggested follow-up queries\n\n"
        "Include specific numbers, dates, and scientific names. Cite data sources.\n\n"
        f"Survey data:\n```json\n{json.dumps(survey_data, indent=2, default=str)}\n```"
    )


@mcp.prompt()
async def species_report(
    scientific_name: str,
    lat: float = 0,
    lon: float = 0,
    radius_km: float = 100,
) -> str:
    """
    Deep dive on a single species: occurrences, climate correlation, audio, soil.

    Searches across all sources for a specific species, optionally scoped to
    a geographic area. Returns occurrence data, environmental context, and
    any available audio recordings.
    """
    import asyncio

    tasks = {
        "search": run_search(
            scientificname=scientific_name,
            lat=lat if lat else None, lon=lon if lon else None,
            radius_km=radius_km if lat else None,
            include_climate=True, limit=30,
            obis=_obis, neon=_neon, era5=_era5,
            inat=_inat, ebird=_ebird,
        ),
        "audio": _xc.search(SearchParams(taxon=scientific_name, limit=5)),
    }

    if lat and lon:
        tasks["soil"] = _soil.search(SearchParams(lat=lat, lng=lon, radius_km=1, limit=1))

    task_keys = list(tasks.keys())
    results_raw = await asyncio.gather(*tasks.values(), return_exceptions=True)
    results = {}
    for key, val in zip(task_keys, results_raw):
        if isinstance(val, Exception):
            results[key] = {"error": str(val)}
        else:
            results[key] = val

    search_data = results.get("search", {})

    audio_data = []
    if isinstance(results.get("audio"), list):
        for obs in results["audio"]:
            audio_data.append({
                "id": obs.id,
                "location": {"lat": obs.location.lat, "lng": obs.location.lng, "country": obs.location.country},
                "recorded_at": obs.observed_at.isoformat() if obs.observed_at else None,
                "audio_url": obs.media_url,
                "quality": obs.quality.grade,
            })

    soil_data = {}
    if isinstance(results.get("soil"), list) and results["soil"]:
        soil_data = results["soil"][0].value or {}

    report_data = {
        "report_type": "species_report",
        "species": scientific_name,
        "search_area": {"lat": lat, "lon": lon, "radius_km": radius_km} if lat else "global",
        "occurrences": search_data.get("species_occurrences", []) if isinstance(search_data, dict) else [],
        "occurrence_count": search_data.get("species_count", 0) if isinstance(search_data, dict) else 0,
        "climate_context": search_data.get("climate") if isinstance(search_data, dict) else None,
        "soil_context": soil_data,
        "audio_recordings": audio_data,
        "neon_sites": search_data.get("neon_sites", []) if isinstance(search_data, dict) else [],
    }

    return (
        f"You are writing a species report for **{scientific_name}**. Below is data "
        "from multiple ecological sources. Synthesize into a species report with:\n\n"
        "1. **Species Overview** — taxonomy, common name, conservation context\n"
        "2. **Distribution & Occurrences** — where observed, spatial patterns, habitat\n"
        "3. **Environmental Associations** — climate conditions at observation sites, "
        "soil properties, elevation range\n"
        "4. **Temporal Patterns** — when observed, seasonal trends, recent vs. historical\n"
        "5. **Audio/Media** — available recordings with links (if any)\n"
        "6. **Monitoring Infrastructure** — nearby NEON sites for long-term tracking\n"
        "7. **Data Sources & Citations** — list all contributing sources with provenance\n\n"
        "Use scientific names and specific data values. Note data quality tiers.\n\n"
        f"Report data:\n```json\n{json.dumps(report_data, indent=2, default=str)}\n```"
    )


@mcp.prompt()
async def site_comparison(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    label1: str = "Site A",
    label2: str = "Site B",
    start_date: str = "",
    end_date: str = "",
) -> str:
    """
    Structured comparison of two locations across all data sources.

    Runs parallel queries for both sites and returns side-by-side data
    on species, climate, soil, and monitoring infrastructure. The agent
    should present a comparative analysis.
    """
    import asyncio
    from datetime import datetime, timedelta

    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    search1, search2, soil1, soil2 = await asyncio.gather(
        run_search(
            lat=lat1, lon=lon1, radius_km=50,
            start_date=start_date, end_date=end_date,
            include_climate=True, limit=20,
            obis=_obis, neon=_neon, era5=_era5,
            inat=_inat, ebird=_ebird,
        ),
        run_search(
            lat=lat2, lon=lon2, radius_km=50,
            start_date=start_date, end_date=end_date,
            include_climate=True, limit=20,
            obis=_obis, neon=_neon, era5=_era5,
            inat=_inat, ebird=_ebird,
        ),
        _soil.search(SearchParams(lat=lat1, lng=lon1, radius_km=1, limit=1)),
        _soil.search(SearchParams(lat=lat2, lng=lon2, radius_km=1, limit=1)),
    )

    comparison_data = {
        "comparison_type": "site_comparison",
        "period": {"start": start_date, "end": end_date},
        "sites": {
            label1: {
                "location": {"lat": lat1, "lon": lon1},
                "species_count": search1.get("species_count", 0),
                "species": search1.get("species_occurrences", []),
                "neon_sites": search1.get("neon_sites", []),
                "climate": search1.get("climate"),
                "soil": soil1[0].value if soil1 else {},
            },
            label2: {
                "location": {"lat": lat2, "lon": lon2},
                "species_count": search2.get("species_count", 0),
                "species": search2.get("species_occurrences", []),
                "neon_sites": search2.get("neon_sites", []),
                "climate": search2.get("climate"),
                "soil": soil2[0].value if soil2 else {},
            },
        },
    }

    return (
        f"You are comparing two ecological sites: **{label1}** vs **{label2}**. "
        "Below is parallel data from multiple sources. Create a structured comparison:\n\n"
        "1. **Location Context** — coordinates, nearest monitoring sites, ecosystem type for each\n"
        "2. **Biodiversity Comparison** — species richness, unique vs. shared species, taxonomic composition\n"
        "3. **Climate Comparison** — temperature, precipitation, seasonality side by side\n"
        "4. **Soil Comparison** — composition, organic carbon, pH differences\n"
        "5. **Monitoring Infrastructure** — NEON coverage, data product availability\n"
        "6. **Key Differences & Similarities** — what distinguishes these sites ecologically?\n"
        "7. **Research Potential** — what questions could a comparative study address here?\n\n"
        "Use tables where helpful. Cite data sources.\n\n"
        f"Comparison data:\n```json\n{json.dumps(comparison_data, indent=2, default=str)}\n```"
    )


@mcp.prompt()
def data_export(
    format: Literal["csv", "geojson", "markdown", "bibtex"] = "markdown",
) -> str:
    """
    Guide the agent through exporting ecological data in a specific format.

    This prompt instructs the agent how to format data from prior queries
    into the requested export format with proper citations and provenance.
    """
    format_instructions = {
        "csv": (
            "Export the ecological data as a CSV file. Include columns:\n"
            "id, scientific_name, common_name, lat, lng, observed_at, source, "
            "quality_tier, license, source_url, relevance_score\n\n"
            "Add a header comment row with the query parameters and date generated. "
            "Use the data from the most recent ecology_search or ecological_survey results "
            "in this conversation."
        ),
        "geojson": (
            "Export the ecological data as a GeoJSON FeatureCollection. Each observation "
            "becomes a Feature with:\n"
            "- geometry: Point with [lon, lat] coordinates\n"
            "- properties: scientific_name, common_name, observed_at, source, quality_tier, "
            "relevance_score, source_url\n\n"
            "Include a top-level 'metadata' property with query parameters and date generated. "
            "Use the data from the most recent ecology_search or ecological_survey results."
        ),
        "markdown": (
            "Export the ecological data as a formatted Markdown report with:\n"
            "1. **Header** — title, date generated, query parameters\n"
            "2. **Summary** — key findings, species count, location, time period\n"
            "3. **Species Table** — markdown table of all observations with key fields\n"
            "4. **Climate Summary** — key climate metrics if available\n"
            "5. **Data Sources** — list of all sources with DOIs and citation strings\n"
            "6. **Methodology** — note that data was federated via Kinship Earth MCP\n\n"
            "Use the data from the most recent ecology_search or ecological_survey results."
        ),
        "bibtex": (
            "Export citation information as BibTeX entries for all data sources used. "
            "Include entries for:\n"
            "- Each data source API (OBIS, NEON, ERA5, eBird, iNaturalist, etc.)\n"
            "- Any datasets with DOIs from the observation provenance\n"
            "- The Kinship Earth MCP server itself\n\n"
            "Use proper BibTeX formatting (@misc, @article, @dataset as appropriate). "
            "Include DOIs, URLs, access dates, and license information."
        ),
    }

    return (
        f"The user wants to export ecological data in **{format}** format.\n\n"
        f"{format_instructions[format]}\n\n"
        "If there is no recent search data in this conversation, first ask the user "
        "what data they want to export and run the appropriate search."
    )


# ---------------------------------------------------------------------------
# Resources — live data for agent discovery
# ---------------------------------------------------------------------------


@mcp.resource("ecology://sources")
async def ecology_sources_resource() -> str:
    """Live registry of all available ecological data sources and their status."""
    result = await run_describe_sources(
        neon=_neon, obis=_obis, era5=_era5,
        inat=_inat, ebird=_ebird,
        gbif=_gbif, nwis=_nwis, xc=_xc,
        movebank=_movebank, fluxnet=_fluxnet,
    )
    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
