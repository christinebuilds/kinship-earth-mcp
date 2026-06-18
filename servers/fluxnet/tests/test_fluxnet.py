"""Tests for the FLUXNET adapter — offline only, no network calls."""

import pytest
from datetime import datetime, timezone

from fluxnet_mcp.adapter import FLUXNETAdapter, KNOWN_SITES
from kinship_shared import SearchParams


@pytest.fixture
def adapter():
    return FLUXNETAdapter(username="", token="")


def test_fluxnet_adapter_id(adapter):
    assert adapter.id == "fluxnet"


def test_fluxnet_capabilities(adapter):
    caps = adapter.capabilities()
    assert caps.adapter_id == "fluxnet"
    assert caps.name == "FLUXNET / AmeriFlux"
    assert caps.supports_location_search is True
    assert caps.supports_taxon_search is False
    assert caps.supports_date_range is True
    assert caps.quality_tier == 1
    assert caps.requires_auth is True
    assert caps.license == "CC-BY-4.0"


def test_fluxnet_capabilities_modality(adapter):
    caps = adapter.capabilities()
    assert "sensor" in caps.modalities


def test_site_to_observation(adapter):
    site = KNOWN_SITES[0]  # Harvard Forest
    obs = adapter._site_to_observation({**site, "distance_km": 0})
    assert obs.modality == "sensor"
    assert obs.location.lat == pytest.approx(site["lat"])
    assert obs.location.lng == pytest.approx(site["lng"])
    assert obs.location.site_id == site["site_id"]
    assert obs.provenance.source_api == "fluxnet"
    assert obs.provenance.doi == "10.1038/s41597-020-0534-3"
    assert obs.provenance.license == "CC-BY-4.0"
    assert obs.temporal_resolution == "30min"
    assert "NEE" in obs.value["measurements"]


@pytest.mark.asyncio
async def test_search_by_location(adapter):
    # Harvard Forest is at 42.54, -72.17 — search within 50km
    params = SearchParams(lat=42.5, lng=-72.2, radius_km=50, limit=10)
    results = await adapter.search(params)
    assert len(results) >= 1
    site_ids = [r.location.site_id for r in results]
    assert "US-Ha1" in site_ids


@pytest.mark.asyncio
async def test_search_by_site_id(adapter):
    params = SearchParams(site_id="US-NR1", limit=5)
    results = await adapter.search(params)
    assert len(results) == 1
    assert results[0].location.site_id == "US-NR1"
    assert results[0].location.site_name == "Niwot Ridge"


@pytest.mark.asyncio
async def test_search_no_location(adapter):
    # Without lat/lng, returns up to limit known sites
    params = SearchParams(limit=5)
    results = await adapter.search(params)
    assert 1 <= len(results) <= 5
    for obs in results:
        assert obs.modality == "sensor"


def test_flux_to_observation_valid(adapter):
    site = KNOWN_SITES[0]
    record = {
        "timestamp": "2023-07-01T00:00:00",
        "NEE_VUT_REF": "-2.5",
        "GPP_NT_VUT_REF": "8.1",
        "LE_F_MDS": "120.0",
    }
    obs = adapter._flux_to_observation(record, {**site, "distance_km": 0})
    assert obs is not None
    assert obs.modality == "sensor"
    assert obs.value["net_ecosystem_exchange_umol_m2_s"] == pytest.approx(-2.5)
    assert obs.value["gross_primary_production_umol_m2_s"] == pytest.approx(8.1)
    assert obs.value["latent_heat_flux_w_m2"] == pytest.approx(120.0)
    assert obs.quality.confidence == pytest.approx(0.95)


def test_flux_to_observation_no_timestamp(adapter):
    site = KNOWN_SITES[0]
    record = {"timestamp": "", "NEE_VUT_REF": "-2.5"}
    obs = adapter._flux_to_observation(record, {**site, "distance_km": 0})
    assert obs is None
