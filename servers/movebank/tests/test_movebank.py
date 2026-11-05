"""Offline tests for the Movebank adapter."""

import pytest
from datetime import datetime, timezone

from movebank_mcp.adapter import MovebankAdapter
from kinship_shared import SearchParams


@pytest.fixture
def adapter():
    return MovebankAdapter()


@pytest.fixture
def adapter_with_auth():
    return MovebankAdapter(username="testuser", password="testpass")


def test_movebank_adapter_id(adapter):
    assert adapter.id == "movebank"


def test_movebank_capabilities(adapter):
    caps = adapter.capabilities()
    assert caps.adapter_id == "movebank"
    assert caps.name == "Movebank"
    assert caps.quality_tier == 1
    assert caps.supports_location_search is True
    assert caps.supports_taxon_search is True
    assert caps.supports_date_range is True
    assert caps.requires_auth is True


def test_movebank_capabilities_modality(adapter):
    caps = adapter.capabilities()
    assert "movement" in caps.modalities


def test_parse_csv_empty(adapter):
    assert adapter._parse_csv("") == []
    assert adapter._parse_csv("No data") == []
    assert adapter._parse_csv("   ") == []


def test_parse_csv_valid(adapter):
    csv_text = "id,name,taxon_ids\n12345,Test Study,Anas platyrhynchos\n"
    result = adapter._parse_csv(csv_text)
    assert len(result) == 1
    assert result[0]["id"] == "12345"
    assert result[0]["name"] == "Test Study"


def test_event_to_observation(adapter):
    event = {
        "location_lat": "42.5",
        "location_long": "-72.1",
        "timestamp": "2024-03-15 10:30:00",
        "individual_local_identifier": "Bird-001",
        "ground_speed": "5.2",
        "heading": "180.0",
    }
    study = {
        "id": "9999",
        "name": "Test Migration Study",
        "taxon_ids": "Anas platyrhynchos",
    }
    obs = adapter._event_to_observation(event, study)
    assert obs is not None
    assert obs.location.lat == 42.5
    assert obs.location.lng == -72.1
    assert obs.modality == "movement"
    assert obs.value["individual_id"] == "Bird-001"
    assert obs.value["speed_m_s"] == 5.2
    assert obs.value["heading_deg"] == 180.0
    assert obs.quality.tier == 1
    assert obs.provenance.source_api == "movebank"


def test_event_to_observation_missing_coords(adapter):
    event = {
        "location_lat": "not_a_number",
        "location_long": "-72.1",
        "timestamp": "2024-03-15 10:30:00",
        "individual_local_identifier": "Bird-001",
    }
    study = {"id": "9999", "name": "Study"}
    obs = adapter._event_to_observation(event, study)
    assert obs is None


@pytest.mark.asyncio
async def test_search_without_auth_returns_empty(adapter):
    """Adapter without credentials returns empty list immediately."""
    params = SearchParams(taxon="Anas platyrhynchos", limit=5)
    results = await adapter.search(params)
    assert results == []
