"""Offline tests for the FLUXNET adapter."""

import pytest
from datetime import datetime, timezone

from fluxnet_mcp.adapter import FLUXNETAdapter, KNOWN_SITES, _haversine_km
from kinship_shared import SearchParams


@pytest.fixture
def adapter():
    return FLUXNETAdapter()


def test_fluxnet_adapter_id(adapter):
    assert adapter.id == "fluxnet"


def test_fluxnet_capabilities(adapter):
    caps = adapter.capabilities()
    assert caps.adapter_id == "fluxnet"
    assert caps.name == "FLUXNET / AmeriFlux"
    assert caps.quality_tier == 1
    assert caps.supports_location_search is True
    assert caps.supports_taxon_search is False
    assert caps.supports_date_range is True
    assert caps.license == "CC-BY-4.0"


def test_fluxnet_capabilities_modality(adapter):
    caps = adapter.capabilities()
    assert "sensor" in caps.modalities


def test_site_to_observation(adapter):
    site = {
        "site_id": "US-Ha1",
        "name": "Harvard Forest",
        "lat": 42.5378,
        "lng": -72.1715,
        "ecosystem": "Deciduous Broadleaf Forest",
        "distance_km": 0,
    }
    obs = adapter._site_to_observation(site)
    assert obs is not None
    assert obs.location.lat == 42.5378
    assert obs.location.lng == -72.1715
    assert obs.location.site_id == "US-Ha1"
    assert obs.location.site_name == "Harvard Forest"
    assert obs.modality == "sensor"
    assert obs.value["site_type"] == "flux_tower"
    assert "NEE" in obs.value["measurements"]
    assert obs.provenance.source_api == "fluxnet"
    assert obs.provenance.license == "CC-BY-4.0"


@pytest.mark.asyncio
async def test_search_by_location(adapter):
    """Search near Harvard Forest coordinates returns US-Ha1."""
    # Harvard Forest is at 42.5378, -72.1715
    params = SearchParams(lat=42.5, lng=-72.2, radius_km=50, limit=5)
    results = await adapter.search(params)
    assert len(results) > 0
    site_ids = [r.location.site_id for r in results]
    assert "US-Ha1" in site_ids


@pytest.mark.asyncio
async def test_search_by_site_id(adapter):
    """Direct site_id lookup finds the site."""
    params = SearchParams(site_id="US-NR1", limit=5)
    results = await adapter.search(params)
    assert len(results) == 1
    assert results[0].location.site_id == "US-NR1"
    assert results[0].location.site_name == "Niwot Ridge"


@pytest.mark.asyncio
async def test_search_no_location(adapter):
    """Without lat/lng, returns all known sites up to limit."""
    params = SearchParams(limit=5)
    results = await adapter.search(params)
    assert len(results) == 5


def test_flux_to_observation_valid(adapter):
    record = {
        "timestamp": "2024-07-01T12:00:00",
        "NEE_VUT_REF": "-2.5",
        "GPP_NT_VUT_REF": "8.3",
        "LE_F_MDS": "150.2",
    }
    site = {
        "site_id": "US-Ha1",
        "name": "Harvard Forest",
        "lat": 42.5378,
        "lng": -72.1715,
        "ecosystem": "Deciduous Broadleaf Forest",
    }
    obs = adapter._flux_to_observation(record, site)
    assert obs is not None
    assert obs.value["net_ecosystem_exchange_umol_m2_s"] == -2.5
    assert obs.value["gross_primary_production_umol_m2_s"] == 8.3
    assert obs.value["latent_heat_flux_w_m2"] == 150.2
    assert obs.modality == "sensor"


def test_flux_to_observation_no_timestamp(adapter):
    """Returns None when timestamp is missing/invalid."""
    record = {
        "timestamp": "not-a-date",
        "NEE_VUT_REF": "-2.5",
    }
    site = {
        "site_id": "US-Ha1",
        "name": "Harvard Forest",
        "lat": 42.5378,
        "lng": -72.1715,
    }
    obs = adapter._flux_to_observation(record, site)
    assert obs is None
