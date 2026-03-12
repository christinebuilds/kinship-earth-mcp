"""
Kinship Earth — Web chat interface.

A FastAPI app that lets users ask ecological questions in plain English.
Claude handles the natural language, the Kinship Earth adapters fetch the data.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from neonscience_mcp.adapter import NeonAdapter
from obis_mcp.adapter import OBISAdapter
from era5_mcp.adapter import ERA5Adapter
from kinship_shared import SearchParams, score_observation

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Kinship Earth")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

claude = anthropic.AsyncAnthropic()

# Initialize adapters (same as the orchestrator)
_neon = NeonAdapter(api_token=os.environ.get("NEON_API_TOKEN"))
_obis = OBISAdapter()
_era5 = ERA5Adapter()

# In-memory conversation store (fine for MVP)
conversations: dict[str, list[dict]] = {}

SYSTEM_PROMPT = """\
You are Kinship Earth, an ecological intelligence assistant that helps \
scientists and curious people explore ecological data using natural language.

You have access to three data sources:
- **OBIS**: 168M+ marine species occurrence records worldwide
- **NEON**: 81 terrestrial ecological monitoring sites across the US
- **ERA5**: Global climate reanalysis data from 1940 to present (temperature, \
precipitation, wind, soil, radiation)

When users ask ecological questions, use your tools to find real data. Always:
- Cite sources with URLs and DOIs when provided in the results
- Present data clearly with context about what it means
- If results are sparse, suggest how to broaden the search
- When showing coordinates, also name the nearest city or region for context
- Format numbers with appropriate units

You can combine data across sources — that's your unique value. For example, \
you can show climate conditions alongside species observations, or find \
monitoring sites near a location of interest.\
"""

# ---------------------------------------------------------------------------
# Tool definitions for Claude API
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "ecology_get_environmental_context",
        "description": (
            "Get the full environmental context for a location and time. "
            "Returns ERA5 climate data (temperature, precipitation, wind, soil) "
            "and nearest NEON monitoring sites. Use this to understand conditions "
            "at a study site or when/where a species was observed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude in decimal degrees"},
                "lon": {"type": "number", "description": "Longitude in decimal degrees (negative = West)"},
                "date": {"type": "string", "description": "Focal date in ISO 8601 format, e.g. '2023-06-15'"},
                "days_before": {
                    "type": "integer",
                    "description": "Days before focal date for climate window (default 7)",
                    "default": 7,
                },
                "days_after": {
                    "type": "integer",
                    "description": "Days after focal date for climate window (default 0)",
                    "default": 0,
                },
            },
            "required": ["lat", "lon", "date"],
        },
    },
    {
        "name": "ecology_search",
        "description": (
            "Unified ecological search across all data sources. Searches OBIS "
            "(marine species), NEON (terrestrial sites), and optionally ERA5 "
            "(climate) simultaneously. Use for species lookups, location-based "
            "discovery, or any ecological data question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scientificname": {
                    "type": "string",
                    "description": "Scientific name, e.g. 'Delphinus delphis'",
                },
                "lat": {"type": "number", "description": "Latitude in decimal degrees"},
                "lon": {"type": "number", "description": "Longitude (negative = West)"},
                "radius_km": {"type": "number", "description": "Search radius in km"},
                "start_date": {"type": "string", "description": "Start date (ISO 8601)"},
                "end_date": {"type": "string", "description": "End date (ISO 8601)"},
                "include_climate": {
                    "type": "boolean",
                    "description": "Include ERA5 climate data (default true)",
                    "default": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max records per source (default 20)",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
    {
        "name": "ecology_describe_sources",
        "description": (
            "Describe all available ecological data sources and their capabilities. "
            "Returns coverage, quality tiers, and what each source can search for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool execution — reuses the orchestrator logic
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta


async def _run_env_context(params: dict) -> dict:
    """Same logic as orchestrator's ecology_get_environmental_context."""
    lat = params["lat"]
    lon = params["lon"]
    date = params["date"]
    days_before = params.get("days_before", 7)
    days_after = params.get("days_after", 0)

    focal = datetime.strptime(date, "%Y-%m-%d")
    start = (focal - timedelta(days=days_before)).strftime("%Y-%m-%d")
    end = (focal + timedelta(days=days_after)).strftime("%Y-%m-%d")

    era5_raw, neon_sites = await asyncio.gather(
        _era5.get_daily(lat=lat, lng=lon, start_date=start, end_date=end),
        _neon.search(SearchParams(lat=lat, lng=lon, radius_km=200, limit=10)),
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
        "query": {"lat": lat, "lon": lon, "focal_date": date, "climate_window": {"start": start, "end": end}},
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
            "provenance": {"doi": "10.24381/cds.adbb2d47", "license": "CC-BY-4.0"},
        },
        "nearby_neon_sites": nearby_neon,
        "nearby_neon_count": len(nearby_neon),
    }


async def _run_search(params: dict) -> dict:
    """Same logic as orchestrator's ecology_search."""
    scientificname = params.get("scientificname")
    lat = params.get("lat")
    lon = params.get("lon")
    radius_km = params.get("radius_km")
    start_date = params.get("start_date")
    end_date = params.get("end_date")
    include_climate = params.get("include_climate", True)
    limit = params.get("limit", 20)

    tasks = {}

    if scientificname or (lat is not None and lon is not None):
        tasks["obis"] = _obis.search(SearchParams(
            taxon=scientificname, lat=lat, lng=lon, radius_km=radius_km,
            start_date=start_date, end_date=end_date, limit=limit,
        ))

    if lat is not None and lon is not None:
        tasks["neon"] = _neon.search(SearchParams(lat=lat, lng=lon, radius_km=radius_km or 200, limit=10))

    if include_climate and lat is not None and lon is not None and start_date and end_date:
        tasks["era5"] = _era5.get_daily(lat=lat, lng=lon, start_date=start_date, end_date=end_date)

    if not tasks:
        return {"error": "Please provide at least a species name or lat/lon coordinates."}

    results = {}
    task_keys = list(tasks.keys())
    settled = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for key, result in zip(task_keys, settled):
        results[key] = {"error": str(result)} if isinstance(result, Exception) else result

    scoring_params = SearchParams(
        taxon=scientificname, lat=lat, lng=lon, radius_km=radius_km,
        start_date=start_date, end_date=end_date, limit=limit,
    )

    obis_occurrences = []
    if "obis" in results and isinstance(results["obis"], list):
        for obs in results["obis"]:
            relevance = score_observation(obs, scoring_params)
            obis_occurrences.append({
                "id": obs.id,
                "scientific_name": obs.taxon.scientific_name if obs.taxon else None,
                "lat": obs.location.lat,
                "lng": obs.location.lng,
                "observed_at": obs.observed_at.isoformat(),
                "basis_of_record": obs.value.get("basis_of_record") if obs.value else None,
                "quality_tier": obs.quality.tier,
                "source_url": obs.provenance.original_url,
                "relevance": {
                    "score": relevance.score,
                    "geo_distance_km": relevance.geo_distance_km,
                    "explanation": relevance.explanation,
                },
            })
        obis_occurrences.sort(key=lambda x: x["relevance"]["score"], reverse=True)

    neon_sites = []
    if "neon" in results and isinstance(results["neon"], list):
        for obs in results["neon"]:
            neon_sites.append({
                "site_code": obs.location.site_id,
                "site_name": obs.location.site_name,
                "lat": obs.location.lat,
                "lng": obs.location.lng,
                "state": obs.location.state_province,
            })

    climate = None
    if "era5" in results and isinstance(results["era5"], dict):
        era5_raw = results["era5"]
        climate = {
            "source": "ERA5 (ECMWF)",
            "daily": era5_raw.get("daily", {}),
            "units": era5_raw.get("daily_units", {}),
        }

    return {
        "species_occurrences": obis_occurrences,
        "species_count": len(obis_occurrences),
        "neon_sites": neon_sites,
        "climate": climate,
    }


async def _run_describe_sources(_params: dict) -> dict:
    """Same logic as orchestrator's ecology_describe_sources."""
    sources = [_neon, _obis, _era5]
    descriptions = []
    for adapter in sources:
        caps = adapter.capabilities()
        descriptions.append({
            "id": caps.adapter_id,
            "name": caps.name,
            "description": caps.description,
            "modalities": caps.modalities,
            "geographic_coverage": caps.geographic_coverage,
            "quality_tier": caps.quality_tier,
            "homepage": caps.homepage_url,
        })
    return {"source_count": len(descriptions), "sources": descriptions}


TOOL_EXECUTORS = {
    "ecology_get_environmental_context": _run_env_context,
    "ecology_search": _run_search,
    "ecology_describe_sources": _run_describe_sources,
}


async def execute_tool(name: str, params: dict) -> str:
    """Execute a tool and return the result as a JSON string."""
    executor = TOOL_EXECUTORS.get(name)
    if not executor:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = await executor(params)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: Request):
    """Streaming chat endpoint. Returns Server-Sent Events."""
    body = await request.json()
    user_message = body.get("message", "")
    conversation_id = body.get("conversation_id", str(uuid.uuid4()))

    # Get or create conversation
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    messages = conversations[conversation_id]

    # Add user message
    messages.append({"role": "user", "content": user_message})

    async def generate():
        nonlocal messages

        yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': conversation_id})}\n\n"

        # Tool use loop — keep calling Claude until it stops requesting tools
        while True:
            # Stream Claude's response
            collected_content = []
            stop_reason = None

            async with claude.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "text":
                            pass  # Text streaming handled by deltas
                        elif event.content_block.type == "tool_use":
                            yield f"data: {json.dumps({'type': 'tool_start', 'tool': event.content_block.name})}\n\n"

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield f"data: {json.dumps({'type': 'text', 'text': event.delta.text})}\n\n"

                final_message = await stream.get_final_message()
                collected_content = final_message.content
                stop_reason = final_message.stop_reason

            # If Claude is done talking, save and exit
            if stop_reason != "tool_use":
                messages.append({"role": "assistant", "content": collected_content})
                break

            # Execute tool calls
            messages.append({"role": "assistant", "content": collected_content})
            tool_results = []

            for block in collected_content:
                if block.type == "tool_use":
                    result_str = await execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': block.name})}\n\n"

            messages.append({"role": "user", "content": tool_results})

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
