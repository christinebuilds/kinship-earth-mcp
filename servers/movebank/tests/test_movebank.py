"""Tests for the Movebank adapter — offline only, no network calls."""

import pytest
from datetime import datetime, timezone

from movebank_mcp.adapter import MovebankAdapter


@pytest.fixture
def adapter():
    return MovebankAdapter(username="", password="")


def test_movebank_adapter_id(adapter):
    assert adapter.id == "movebank"


def test_movebank_capabilities(adapter):
    caps = adapter.capabilities()
    assert caps.adapter_id == "movebank"
    assert caps.name == "Movebank"
    assert caps.supports_location_search is True
    assert caps.supports_taxon_search is True
    assert caps.supports_date_range is True
    assert caps.quality_tier == 1
    assert caps.requires_auth is True


def test_movebank_capabilities_modality(adapter):
    caps = adapter.capabilities()
    assert "movement" in caps.modalities


def test_parse_csv_empty(adapter):
    assert adapter._parse_csv("") == []
    assert adapter._parse_csv("   ") == []
    assert adapter._parse_csv("No data") == []


def test_parse_csv_valid(adapter):
    csv_text = "id,name,taxon_ids\n1,Study A,Panthera leo\n2,Study B,Aquila chrysaetos\n"
    result = adapter._parse_csv(csv_text)
    assert len(result) == 2
    assert result[0]["id"] == "1"
    assert result[0]["name"] == "Study A"
    assert result[1]["taxon_ids"] == "Aquila chrysaetos"


def test_event_to_observation(adapter):
    event = {
        "location_lat": "42.37",
        "location_long": "-71.12",
        "timestamp": "2023-06-15 12:30:00",
        "individual_local_identifier": "tag-001",
        "ground_speed": "5.2",
        "heading": "180.0",
    }
    study = {"id": "12345", "name": "Test Study", "taxon_ids": "Accipiter cooperii"}
    obs = adapter._event_to_observation(event, study)
    assert obs is not None
    assert obs.modality == "movement"
    assert obs.location.lat == pytest.approx(42.37)
    assert obs.location.lng == pytest.approx(-71.12)
    assert obs.taxon.scientific_name == "Accipiter cooperii"
    assert obs.value["individual_id"] == "tag-001"
    assert obs.value["speed_m_s"] == pytest.approx(5.2)
    assert obs.value["heading_deg"] == pytest.approx(180.0)
    assert obs.provenance.source_api == "movebank"
    assert obs.quality.tier == 1


def test_event_to_observation_missing_coords(adapter):
    event = {
        "location_lat": "",
        "location_long": "",
        "timestamp": "2023-06-15 12:30:00",
        "individual_local_identifier": "tag-001",
    }
    study = {"id": "12345", "name": "Test Study", "taxon_ids": ""}
    obs = adapter._event_to_observation(event, study)
    assert obs is None


@pytest.mark.asyncio
async def test_search_without_auth_returns_empty():
    adapter = MovebankAdapter(username="", password="")
    from kinship_shared import SearchParams
    params = SearchParams(taxon="Anas platyrhynchos", limit=5)
    results = await adapter.search(params)
    assert results == []
