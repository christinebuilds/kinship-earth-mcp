"""
Evaluation Test: ecology_whats_around_me — Citizen Discovery Tool
=================================================================

PERSONA: Maria Gonzalez — Backyard Naturalist
Question: "What's happening in the ecosystem around my home in San Francisco?"

This is the citizen-facing entry point to Kinship Earth. Unlike the research
tools, it combines recent species sightings, monitoring sites, and climate
into a single friendly snapshot. It must work for any location, handle
data-sparse regions gracefully, and return quickly (< 15s).

RUNNING THESE TESTS
-------------------
From the repo root:
  uv run --package kinship-orchestrator pytest servers/orchestrator/tests/test_whats_around_me.py -v
"""

import pytest

from kinship_orchestrator.server import ecology_whats_around_me


# Well-known test locations
SF_LAT, SF_LON = 37.77, -122.47       # Golden Gate Park, San Francisco
MONTEREY_LAT, MONTEREY_LON = 36.6, -121.9  # Monterey Bay, CA


# ===========================================================================
# Layer 1: Connectivity — does the tool work at all?
# ===========================================================================

class TestConnectivity:

    @pytest.mark.asyncio
    async def test_returns_results_for_sf(self):
        """Canary test: SF Bay Area should always have ecological data."""
        result = await ecology_whats_around_me(lat=SF_LAT, lon=SF_LON)
        assert isinstance(result, dict)
        assert "snapshot" in result, "Response must include snapshot summary"

    @pytest.mark.asyncio
    async def test_handles_ocean_gracefully(self):
        """Mid-ocean query should return empty snapshot, not crash."""
        result = await ecology_whats_around_me(lat=0.0, lon=0.0, radius_km=10)
        assert isinstance(result, dict)
        assert "snapshot" in result
        # May have zero results but should not error


# ===========================================================================
# Layer 2: Contract — does the response have the right structure?
# ===========================================================================

class TestContract:

    @pytest.mark.asyncio
    async def test_response_has_required_keys(self):
        """The response must include all top-level keys."""
        result = await ecology_whats_around_me(lat=SF_LAT, lon=SF_LON)

        assert "snapshot" in result
        assert "recent_sightings" in result
        assert "neon_sites" in result
        assert "climate" in result
        assert "sources_queried" in result

    @pytest.mark.asyncio
    async def test_snapshot_has_summary_fields(self):
        """The snapshot section summarizes what was found."""
        result = await ecology_whats_around_me(lat=SF_LAT, lon=SF_LON)
        snapshot = result["snapshot"]

        assert "location" in snapshot
        assert "lat" in snapshot["location"]
        assert "lon" in snapshot["location"]
        assert "radius_km" in snapshot["location"]
        assert "period" in snapshot
        assert "unique_species" in snapshot
        assert "total_observations" in snapshot
        assert "monitoring_sites_nearby" in snapshot
        assert "climate_data_available" in snapshot

    @pytest.mark.asyncio
    async def test_snapshot_types_are_correct(self):
        """Summary fields have the right types for display."""
        result = await ecology_whats_around_me(lat=SF_LAT, lon=SF_LON)
        snapshot = result["snapshot"]

        assert isinstance(snapshot["unique_species"], int)
        assert isinstance(snapshot["total_observations"], int)
        assert isinstance(snapshot["monitoring_sites_nearby"], int)
        assert isinstance(snapshot["climate_data_available"], bool)
        assert isinstance(snapshot["period"], str)

    @pytest.mark.asyncio
    async def test_multiple_sources_queried(self):
        """Should query at least 2 different data sources."""
        result = await ecology_whats_around_me(lat=SF_LAT, lon=SF_LON)
        sources = result.get("sources_queried", [])
        assert len(sources) >= 2, (
            f"Should query multiple sources, got: {sources}"
        )

    @pytest.mark.asyncio
    async def test_recent_sightings_capped_at_20(self):
        """Recent sightings should be capped for readability."""
        result = await ecology_whats_around_me(lat=SF_LAT, lon=SF_LON)
        sightings = result.get("recent_sightings", [])
        assert len(sightings) <= 20, (
            f"Recent sightings should be capped at 20, got {len(sightings)}"
        )


# ===========================================================================
# Layer 3: Semantic — is the data correct?
# ===========================================================================

class TestSemantic:

    @pytest.mark.network
    @pytest.mark.asyncio
    async def test_sf_has_observations(self):
        """San Francisco should always have ecological observations."""
        result = await ecology_whats_around_me(lat=SF_LAT, lon=SF_LON)
        snapshot = result["snapshot"]

        assert snapshot["total_observations"] > 0, (
            "SF Bay Area should have species observations within 25km / 7 days"
        )

    @pytest.mark.asyncio
    async def test_default_radius_is_25km(self):
        """Default radius should be 25km."""
        result = await ecology_whats_around_me(lat=SF_LAT, lon=SF_LON)
        assert result["snapshot"]["location"]["radius_km"] == 25

    @pytest.mark.asyncio
    async def test_default_period_is_7_days(self):
        """Default period should cover last 7 days."""
        result = await ecology_whats_around_me(lat=SF_LAT, lon=SF_LON)
        assert "7 days" in result["snapshot"]["period"]


# ===========================================================================
# Layer 4: Scientific fitness — provenance and quality
# ===========================================================================

class TestScientificFitness:

    @pytest.mark.asyncio
    async def test_sightings_have_provenance(self):
        """Every sighting should be traceable to its data source."""
        result = await ecology_whats_around_me(lat=SF_LAT, lon=SF_LON)
        sightings = result.get("recent_sightings", [])

        if len(sightings) > 0:
            for s in sightings[:5]:  # spot-check first 5
                assert "source" in s or "source_api" in s, (
                    f"Sighting missing source attribution: {list(s.keys())}"
                )


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_small_radius(self):
        """Very small radius should work but may be empty."""
        result = await ecology_whats_around_me(
            lat=SF_LAT, lon=SF_LON, radius_km=0.5,
        )
        assert isinstance(result, dict)
        assert "snapshot" in result

    @pytest.mark.asyncio
    async def test_large_radius(self):
        """Large radius should not crash or timeout."""
        result = await ecology_whats_around_me(
            lat=SF_LAT, lon=SF_LON, radius_km=200,
        )
        assert isinstance(result, dict)
        assert "snapshot" in result

    @pytest.mark.asyncio
    async def test_arctic_coordinates(self):
        """Arctic location — sparse data, should not crash."""
        result = await ecology_whats_around_me(
            lat=71.0, lon=-156.8,  # Utqiagvik, Alaska
            radius_km=50,
        )
        assert isinstance(result, dict)
        assert "snapshot" in result
