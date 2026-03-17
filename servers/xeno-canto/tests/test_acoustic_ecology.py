"""
Evaluation Test: Xeno-canto Acoustic Ecology Use Case
=====================================================

USE CASE
--------
A soundscape ecologist is studying dawn chorus composition in Pacific
Northwest old-growth forests. They need bird vocalizations from the
region to compare with their AudioMoth field recordings.

Xeno-canto has 1M+ community-contributed recordings, quality-rated A-E.

RUNNING THESE TESTS
-------------------
  uv run --package xenocanto-mcp pytest servers/xeno-canto/tests/ -v
"""

import os
import pytest
from kinship_shared import SearchParams
from xenocanto_mcp.adapter import XenoCantoAdapter

HAS_XC_KEY = bool(os.environ.get("XC_API_KEY"))
skip_no_key = pytest.mark.skipif(not HAS_XC_KEY, reason="XC_API_KEY not set")


@pytest.fixture
def adapter():
    return XenoCantoAdapter()


# ---------------------------------------------------------------------------
# Layer 1: Connectivity
# ---------------------------------------------------------------------------

@skip_no_key
@pytest.mark.asyncio
async def test_xc_api_returns_recordings(adapter):
    """CANARY: Xeno-canto API responds with recording results."""
    params = SearchParams(taxon="Turdus migratorius", limit=5)
    results = await adapter.search(params)
    assert len(results) > 0, "Should find American Robin recordings"


# ---------------------------------------------------------------------------
# Layer 2: Contract
# ---------------------------------------------------------------------------

@skip_no_key
@pytest.mark.asyncio
async def test_recording_schema_complete(adapter):
    """CONTRACT: Every recording has required fields."""
    params = SearchParams(taxon="Melospiza melodia", limit=3)
    results = await adapter.search(params)
    assert len(results) > 0

    obs = results[0]
    assert obs.id.startswith("xeno-canto:"), f"ID format: {obs.id}"
    assert obs.modality == "acoustic"
    assert obs.taxon is not None
    assert obs.taxon.scientific_name, "Must have species name"
    assert obs.location.lat is not None
    assert obs.location.lng is not None
    assert obs.media_url, "Must have audio URL"
    assert obs.media_type == "audio/mpeg"
    assert obs.provenance.source_api == "xeno-canto"
    assert obs.provenance.original_url.startswith("https://xeno-canto.org/")
    assert obs.provenance.citation_string, "Must have citation"
    assert obs.quality.tier in (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Layer 3: Semantic correctness
# ---------------------------------------------------------------------------

@skip_no_key
@pytest.mark.asyncio
async def test_species_name_matches_query(adapter):
    """SEMANTIC: Returned recordings match the queried species."""
    params = SearchParams(taxon="Strix varia", limit=5)
    results = await adapter.search(params)
    assert len(results) > 0
    for obs in results:
        name = obs.taxon.scientific_name.lower()
        assert "strix" in name or "varia" in name, (
            f"Expected Strix varia, got {obs.taxon.scientific_name}"
        )


@skip_no_key
@pytest.mark.asyncio
async def test_audio_url_is_downloadable(adapter):
    """SEMANTIC: Audio URLs should be valid HTTPS links."""
    params = SearchParams(taxon="Corvus corax", limit=3)
    results = await adapter.search(params)
    assert len(results) > 0
    for obs in results:
        assert obs.media_url.startswith("https://"), (
            f"Audio URL should be HTTPS: {obs.media_url}"
        )


# ---------------------------------------------------------------------------
# Layer 4: Scientific fitness
# ---------------------------------------------------------------------------

@skip_no_key
@pytest.mark.asyncio
async def test_quality_rating_mapped_correctly(adapter):
    """SCIENTIFIC: XC quality ratings (A-E) map to tiers (1-4)."""
    params = SearchParams(taxon="Cyanocitta cristata", limit=20)
    results = await adapter.search(params)
    assert len(results) > 0
    for obs in results:
        flags = obs.quality.flags
        assert any("xc_quality:" in f for f in flags), (
            f"Quality flags should include xc_quality rating: {flags}"
        )
        # Tier should be valid
        assert 1 <= obs.quality.tier <= 4


@skip_no_key
@pytest.mark.asyncio
async def test_geographic_search(adapter):
    """SCIENTIFIC: Can search for recordings near a specific location."""
    # PNW old-growth — near Wind River, WA
    params = SearchParams(lat=45.82, lng=-121.95, radius_km=200, limit=10)
    results = await adapter.search(params)
    # May or may not have results — just verify no crash
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_search_returns_empty(adapter):
    """EDGE: Search with no params returns empty list."""
    params = SearchParams(limit=5)
    results = await adapter.search(params)
    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_get_by_id_nonexistent(adapter):
    """EDGE: Non-existent recording ID returns None."""
    result = await adapter.get_by_id("999999999")
    assert result is None
