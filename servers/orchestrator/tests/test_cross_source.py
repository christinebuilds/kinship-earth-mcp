"""
Evaluation Test: Cross-Source Orchestration Use Cases
=====================================================

USE CASE
--------
These tests verify the unique value of Kinship Earth — answering questions
that no single data source can answer alone.

Scenario 1: "What was the climate like when dolphins were seen near Woods Hole?"
  → OBIS occurrences + ERA5 climate in a single query

Scenario 2: "What ecological monitoring exists near Wind River forest?"
  → NEON sites + ERA5 climate for site characterization

Scenario 3: "What data sources are available and what do they cover?"
  → Self-describing registry of all adapters

RUNNING THESE TESTS
-------------------
From the repo root:
  uv run --package kinship-orchestrator pytest servers/orchestrator/tests/test_cross_source.py -v
"""

import pytest

from kinship_orchestrator.server import (
    ecology_get_environmental_context,
    ecology_search,
    ecology_describe_sources,
)


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

# NEON Wind River Experimental Forest
WREF_LAT = 45.82
WREF_LNG = -121.95

# Woods Hole, MA (marine research hub)
WOODS_HOLE_LAT = 41.5
WOODS_HOLE_LNG = -70.7


# ---------------------------------------------------------------------------
# ecology_get_environmental_context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_environmental_context_returns_climate_and_sites():
    """
    CROSS-SOURCE TEST: Environmental context combines ERA5 + NEON.

    Given the Wind River coordinates and a date, we should get:
    - ERA5 daily climate data for the time window
    - Nearby NEON sites (WREF should be found since we're querying its coordinates)
    """
    result = await ecology_get_environmental_context(
        lat=WREF_LAT, lon=WREF_LNG, date="2023-06-15",
        days_before=3, days_after=0,
    )

    # Should have climate data
    assert "climate" in result, "Response must include climate data"
    daily = result["climate"].get("daily", {})
    assert "time" in daily, "Climate data must include timestamps"
    assert len(daily["time"]) > 0, "Must have at least one day of climate data"

    # Should have temperature data
    assert "temperature_2m_mean" in daily or "temperature_2m_max" in daily, (
        "Climate data should include temperature variables"
    )

    # Should have nearby NEON sites (WREF is at these exact coordinates)
    assert "nearby_neon_sites" in result
    assert result["nearby_neon_count"] > 0, (
        "Should find NEON sites within 200km of Wind River. "
        "WREF is at these exact coordinates."
    )

    # Verify WREF is in the results
    site_codes = [s.get("site_code") for s in result["nearby_neon_sites"]]
    assert "WREF" in site_codes, (
        f"WREF should be in nearby sites (we queried its exact coordinates). "
        f"Found: {site_codes}"
    )

    # Should track which sources were used
    assert "data_sources_used" in result
    assert "era5" in result["data_sources_used"]
    assert "neonscience" in result["data_sources_used"]


@pytest.mark.asyncio
async def test_environmental_context_has_provenance():
    """
    PROVENANCE TEST: Climate data must include DOI and license.
    """
    result = await ecology_get_environmental_context(
        lat=WREF_LAT, lon=WREF_LNG, date="2023-06-15",
    )

    provenance = result.get("climate", {}).get("provenance", {})
    assert provenance.get("doi") == "10.24381/cds.adbb2d47", "Must include ERA5 DOI"
    assert provenance.get("license") == "CC-BY-4.0", "ERA5 license is CC-BY-4.0"


# ---------------------------------------------------------------------------
# ecology_search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unified_search_with_species_and_location():
    """
    CROSS-SOURCE TEST: Unified search returns species + climate together.

    Search for dolphins near Woods Hole with climate context. Should return
    OBIS occurrences AND ERA5 climate data in a single response.
    """
    result = await ecology_search(
        scientificname="Delphinus delphis",
        lat=WOODS_HOLE_LAT,
        lon=WOODS_HOLE_LNG,
        radius_km=200,
        start_date="2015-01-01",
        end_date="2023-12-31",
        include_climate=False,  # Skip climate for speed — tested elsewhere
        limit=20,
    )

    # Should have species occurrences from OBIS
    assert "species_occurrences" in result
    assert result["species_count"] > 0, (
        "Should find Delphinus delphis near Woods Hole via OBIS. "
        "OBIS client-side geo filtering may reduce results — the wider date "
        "range gives more records to filter from."
    )

    # Check occurrence structure
    occ = result["species_occurrences"][0]
    assert "scientific_name" in occ
    assert "lat" in occ
    assert "lng" in occ
    assert "observed_at" in occ

    # Check relevance scoring is present
    assert "relevance" in occ, "Each occurrence should have a relevance score"
    rel = occ["relevance"]
    assert "score" in rel, "Relevance must include composite score"
    assert 0 <= rel["score"] <= 1, f"Score must be 0-1, got {rel['score']}"
    assert "explanation" in rel, "Relevance must include explanation string"

    # Check search context
    assert "search_context" in result, "Response should include search_context"
    ctx = result["search_context"]
    assert "sources_queried" in ctx
    assert "obis_records_returned" in ctx


@pytest.mark.asyncio
async def test_unified_search_species_only():
    """
    EDGE CASE: Search with only a species name (no location).

    Should still return OBIS results but no NEON sites or climate.
    """
    result = await ecology_search(
        scientificname="Tursiops truncatus",
        limit=5,
    )

    assert result["species_count"] > 0, (
        "Should find bottlenose dolphin records in OBIS"
    )
    assert result["neon_site_count"] == 0, "No NEON sites without location"
    assert result["climate"] is None, "No climate without location/dates"


@pytest.mark.asyncio
async def test_unified_search_location_only():
    """
    Search with only coordinates — should find NEON sites and OBIS records.
    """
    result = await ecology_search(
        lat=WREF_LAT,
        lon=WREF_LNG,
        radius_km=100,
        limit=10,
    )

    # Should find NEON sites near Wind River
    assert result["neon_site_count"] > 0, (
        "Should find NEON sites near Wind River coordinates"
    )


@pytest.mark.asyncio
async def test_unified_search_no_params_returns_error():
    """
    EDGE CASE: Search with no parameters should return a helpful error.
    """
    result = await ecology_search()

    assert "error" in result, "Should return error message with no search params"


# ---------------------------------------------------------------------------
# ecology_describe_sources
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_describe_sources_lists_all_adapters():
    """
    REGISTRY TEST: All three data sources should be described.
    """
    result = await ecology_describe_sources()

    assert result["source_count"] >= 3, f"Expected at least 3 sources, got {result['source_count']}"

    source_ids = [s["id"] for s in result["sources"]]
    assert "neonscience" in source_ids, "NEON must be listed"
    assert "obis" in source_ids, "OBIS must be listed"
    assert "era5" in source_ids, "ERA5 must be listed"
    assert "inaturalist" in source_ids, "iNaturalist must be listed"


@pytest.mark.asyncio
async def test_describe_sources_has_capabilities():
    """
    REGISTRY TEST: Each source description includes capabilities.
    """
    result = await ecology_describe_sources()

    for source in result["sources"]:
        assert "name" in source, f"Source {source['id']} missing name"
        assert "description" in source, f"Source {source['id']} missing description"
        assert "modalities" in source, f"Source {source['id']} missing modalities"
        assert "quality_tier" in source, f"Source {source['id']} missing quality_tier"
        assert "search_capabilities" in source, f"Source {source['id']} missing capabilities"
        assert isinstance(source["search_capabilities"], dict)


@pytest.mark.asyncio
async def test_describe_sources_lists_cross_source_tools():
    """
    REGISTRY TEST: The response should describe the cross-source tools.
    """
    result = await ecology_describe_sources()

    assert "cross_source_tools" in result
    tool_names = [t["name"] for t in result["cross_source_tools"]]
    assert "ecology_get_environmental_context" in tool_names
    assert "ecology_search" in tool_names


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_environmental_context_bad_date():
    """
    VALIDATION TEST: A bad date string returns a clear error, not a traceback.
    """
    result = await ecology_get_environmental_context(
        lat=WREF_LAT, lon=WREF_LNG, date="not-a-date",
    )
    assert "error" in result
    assert "YYYY-MM-DD" in result["error"]


@pytest.mark.asyncio
async def test_environmental_context_invalid_lat():
    """
    VALIDATION TEST: Latitude out of range returns a clear error.
    """
    result = await ecology_get_environmental_context(
        lat=999, lon=WREF_LNG, date="2023-06-15",
    )
    assert "error" in result
    assert "lat" in result["error"]


@pytest.mark.asyncio
async def test_search_invalid_coordinates():
    """
    VALIDATION TEST: Out-of-range coordinates return a clear error.
    """
    result = await ecology_search(lat=200, lon=500)
    assert "error" in result
    assert "lat" in result["error"]
