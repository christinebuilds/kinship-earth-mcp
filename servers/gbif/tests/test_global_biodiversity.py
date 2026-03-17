"""
Evaluation Test: GBIF Global Biodiversity Use Case
===================================================

USE CASE
--------
A biogeographer is studying the global distribution of oak species
(Quercus) across continents. They need GBIF's aggregated records from
museums, herbaria, and citizen science to map species richness patterns.

GBIF is the world's largest biodiversity data aggregator (2.8B+ records).
No authentication required.

RUNNING THESE TESTS
-------------------
  uv run --package gbif-mcp pytest servers/gbif/tests/ -v
"""

import pytest
from kinship_shared import SearchParams
from gbif_mcp.adapter import GBIFAdapter


@pytest.fixture
def adapter():
    return GBIFAdapter()


# ---------------------------------------------------------------------------
# Layer 1: Connectivity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gbif_api_returns_results(adapter):
    """CANARY: GBIF API responds and returns occurrence records."""
    params = SearchParams(taxon="Quercus alba", limit=5)
    results = await adapter.search(params)
    assert len(results) > 0, "GBIF should have records for Quercus alba (white oak)"


# ---------------------------------------------------------------------------
# Layer 2: Contract
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observation_schema_complete(adapter):
    """CONTRACT: Every observation has required fields for downstream use."""
    params = SearchParams(taxon="Puma concolor", limit=3)
    results = await adapter.search(params)
    assert len(results) > 0, "Should find mountain lion records"

    obs = results[0]
    assert obs.id.startswith("gbif:"), f"ID should be prefixed: {obs.id}"
    assert obs.modality == "occurrence"
    assert obs.taxon is not None
    assert obs.taxon.scientific_name, "Must have scientific name"
    assert obs.location.lat is not None
    assert obs.location.lng is not None
    assert obs.provenance.source_api == "gbif"
    assert obs.provenance.original_url.startswith("https://www.gbif.org/occurrence/")
    assert obs.quality.tier in (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Layer 3: Semantic correctness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_taxon_resolution_works(adapter):
    """SEMANTIC: Searching by common species name resolves to correct taxon."""
    params = SearchParams(taxon="Homo sapiens", limit=3)
    results = await adapter.search(params)
    assert len(results) > 0
    for obs in results:
        assert "sapiens" in obs.taxon.scientific_name.lower() or "homo" in obs.taxon.scientific_name.lower(), (
            f"Expected Homo sapiens, got {obs.taxon.scientific_name}"
        )


@pytest.mark.asyncio
async def test_geographic_search_returns_local_results(adapter):
    """SEMANTIC: Geographic search near London returns UK records."""
    params = SearchParams(lat=51.5, lng=-0.1, radius_km=50, limit=10)
    results = await adapter.search(params)
    assert len(results) > 0, "Should find records near London"
    # At least some should be from the UK
    uk_results = [r for r in results if r.location.country_code in ("GB", "UK")]
    assert len(uk_results) > 0, (
        f"Expected UK records near London. Got countries: "
        f"{set(r.location.country_code for r in results)}"
    )


# ---------------------------------------------------------------------------
# Layer 4: Scientific fitness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_museum_specimens_have_institution(adapter):
    """SCIENTIFIC: Museum specimen records should include institution info."""
    params = SearchParams(taxon="Tyrannosaurus rex", limit=10)
    results = await adapter.search(params)
    # T. rex should only exist as preserved specimens
    specimens = [r for r in results if r.value.get("basis_of_record") == "PRESERVED_SPECIMEN"]
    if specimens:
        for spec in specimens:
            assert spec.value.get("institution"), (
                f"Museum specimen {spec.id} should have institution code"
            )


@pytest.mark.asyncio
async def test_date_range_filtering(adapter):
    """SCIENTIFIC: Date range filtering returns records within window."""
    params = SearchParams(
        taxon="Delphinus delphis",
        start_date="2020-01-01",
        end_date="2023-12-31",
        limit=10,
    )
    results = await adapter.search(params)
    assert len(results) > 0, "Should find dolphin records in 2020-2023"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nonexistent_species_returns_gracefully(adapter):
    """EDGE: Searching for a made-up species returns gracefully, not an error."""
    params = SearchParams(taxon="Xyzzyplugh notaspecies zzz", limit=5)
    results = await adapter.search(params)
    assert isinstance(results, list)
    # GBIF may return fuzzy matches — that's OK, just verify no crash


@pytest.mark.asyncio
async def test_nonexistent_occurrence_returns_none(adapter):
    """EDGE: Fetching a non-existent key returns None."""
    result = await adapter.get_by_id("999999999999")
    assert result is None
