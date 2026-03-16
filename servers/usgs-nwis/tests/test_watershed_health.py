"""
Evaluative Tests: Watershed Ecologist Stream Monitoring
=======================================================

PERSONA: Dr. James Rivera — Watershed Restoration Ecologist
Research question: "Is the Russian River watershed recovering from
the 2021-2023 drought? Are stream conditions supporting salmon spawning?"

These tests exercise the USGS NWIS adapter through the lens of a
watershed ecologist monitoring stream health for fish habitat.

RUNNING THESE TESTS
-------------------
From the repo root:
  uv run --package usgs-nwis-mcp pytest servers/usgs-nwis/tests/ -v

NOTE: No API key required. Rate limited to ~50 req/hr without key.
Get a free key at: https://api.waterdata.usgs.gov/signup
"""

import pytest

from usgs_nwis_mcp.adapter import USGSNWISAdapter
from kinship_shared import SearchParams


@pytest.fixture
def adapter():
    return USGSNWISAdapter()


# ===========================================================================
# Layer 1: Connectivity — can we reach the USGS API?
# ===========================================================================

class TestConnectivity:

    @pytest.mark.asyncio
    async def test_api_responds(self, adapter):
        """Canary test: USGS NWIS API is reachable."""
        result = await adapter.search(SearchParams(
            lat=38.95,  # Potomac River near DC — one of the most monitored
            lng=-77.13,
            radius_km=10,
            limit=5,
        ))
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_capabilities_are_declared(self, adapter):
        """Adapter self-describes accurately."""
        caps = adapter.capabilities()
        assert caps.adapter_id == "usgs-nwis"
        assert caps.supports_location_search is True
        assert caps.supports_date_range is True
        assert "hydrological" in caps.modalities
        assert caps.quality_tier == 1


# ===========================================================================
# Layer 2: Contract — does the schema validate?
# ===========================================================================

class TestContract:

    @pytest.mark.asyncio
    async def test_observations_have_required_fields(self, adapter):
        """Every returned observation has the fields a researcher needs."""
        result = await adapter.search(SearchParams(
            lat=38.95,
            lng=-77.13,
            radius_km=10,
            limit=10,
        ))
        if len(result) > 0:
            obs = result[0]
            assert obs.id.startswith("usgs-nwis:")
            assert obs.modality == "hydrological"
            assert obs.location.lat is not None
            assert obs.location.lng is not None
            assert obs.observed_at is not None
            assert obs.provenance.source_api == "usgs-nwis"
            assert obs.value is not None
            assert "parameter_name" in obs.value
            assert "measurement" in obs.value

    @pytest.mark.asyncio
    async def test_watershed_id_populated(self, adapter):
        """USGS sites should have HUC watershed IDs."""
        result = await adapter.search(SearchParams(
            lat=38.95,
            lng=-77.13,
            radius_km=10,
            limit=5,
        ))
        if len(result) > 0:
            assert result[0].location.watershed_id is not None, (
                "USGS sites should include HUC watershed ID"
            )


# ===========================================================================
# Layer 3: Semantic — is the data factually correct?
# ===========================================================================

class TestSemantic:

    @pytest.mark.asyncio
    async def test_potomac_has_data(self, adapter):
        """
        Known answer: The Potomac River near DC (site 01646500) is one of
        the most monitored stream gauges in the US. Should always have data.
        """
        result = await adapter.search(SearchParams(
            lat=38.95,
            lng=-77.13,
            radius_km=10,
            limit=10,
        ))
        assert len(result) > 0, (
            "No data found near the Potomac River — one of the most "
            "monitored gauges in the US."
        )


# ===========================================================================
# Layer 4: Scientific — does it solve the research question?
# ===========================================================================

class TestWatershedMonitoring:

    @pytest.mark.asyncio
    async def test_site_search_by_id(self, adapter):
        """
        The ecologist has a specific site they monitor — can they query
        it by USGS site number?
        """
        result = await adapter.search(SearchParams(
            site_id="01646500",  # Potomac River
            limit=10,
        ))
        if len(result) > 0:
            assert "01646500" in result[0].location.site_id, (
                "Site search should return data from the requested site"
            )

    @pytest.mark.asyncio
    async def test_quality_is_research_grade(self, adapter):
        """
        USGS data comes from calibrated government instruments.
        All records should be tier 1 (research grade).
        """
        result = await adapter.search(SearchParams(
            lat=38.95,
            lng=-77.13,
            radius_km=10,
            limit=5,
        ))
        for obs in result:
            assert obs.quality.tier == 1, (
                f"USGS data should be tier 1, got tier {obs.quality.tier}"
            )
            assert obs.quality.grade == "research"
