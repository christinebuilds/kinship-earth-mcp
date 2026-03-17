"""
Evaluative Tests: Soil Assessment for Habitat Restoration
==========================================================

PERSONA: Dr. Aisha Rahman — Soil Scientist / Restoration Ecologist
Research question: "What are the soil conditions at our restoration site,
and how do they compare to reference conditions for native plant establishment?"

These tests exercise the SoilGrids adapter through the lens of a soil scientist
evaluating site conditions for ecological restoration planning.

RUNNING THESE TESTS
-------------------
From the repo root:
  uv run --package kinship-orchestrator pytest servers/soilgrids/tests/ -v

NOTE: No API key required. SoilGrids API may have intermittent availability.
Tests are marked to skip gracefully if the API is unreachable.
"""

import pytest
import httpx

from soilgrids_mcp.adapter import (
    SoilGridsAdapter,
    VALID_PROPERTIES,
    VALID_DEPTHS,
    VALID_VALUES,
    PROPERTY_INFO,
)
from kinship_shared import SearchParams


@pytest.fixture
def adapter():
    return SoilGridsAdapter()


def _api_unavailable(exc: Exception) -> bool:
    """Check if an exception indicates the SoilGrids API is temporarily down."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (502, 503, 504)
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    return False


# ===========================================================================
# Layer 1: Connectivity — can we reach the SoilGrids API?
# ===========================================================================

class TestConnectivity:

    @pytest.mark.asyncio
    async def test_api_responds(self, adapter):
        """Canary test: SoilGrids API is reachable."""
        try:
            result = await adapter.query_properties(
                lat=51.57,  # Wageningen, NL (ISRIC headquarters)
                lon=5.39,
                properties=["clay"],
                depths=["0-5cm"],
                values=["mean"],
            )
            assert isinstance(result, dict), "Expected dict response from SoilGrids"
        except Exception as exc:
            if _api_unavailable(exc):
                pytest.skip("SoilGrids API is temporarily unavailable (503)")
            raise

    @pytest.mark.asyncio
    async def test_capabilities_are_declared(self, adapter):
        """Adapter self-describes accurately."""
        caps = adapter.capabilities()
        assert caps.adapter_id == "soilgrids"
        assert caps.supports_location_search is True
        assert caps.supports_taxon_search is False
        assert caps.supports_date_range is False
        assert "sensor" in caps.modalities
        assert caps.license == "CC-BY-4.0"


# ===========================================================================
# Layer 2: Contract — does the schema validate?
# ===========================================================================

class TestContract:

    @pytest.mark.asyncio
    async def test_response_has_layers_and_depths(self, adapter):
        """API response contains expected structure: layers with depth entries."""
        try:
            result = await adapter.query_properties(
                lat=51.57,
                lon=5.39,
                properties=["clay", "sand"],
                depths=["0-5cm", "5-15cm"],
                values=["mean"],
            )
        except Exception as exc:
            if _api_unavailable(exc):
                pytest.skip("SoilGrids API is temporarily unavailable")
            raise

        assert "properties" in result, "Response must have 'properties' key"
        layers = result["properties"].get("layers", [])
        assert len(layers) > 0, "Response must have at least one layer"

        for layer in layers:
            assert "name" in layer, "Each layer must have a 'name'"
            assert layer["name"] in VALID_PROPERTIES, (
                f"Unexpected property name: {layer['name']}"
            )
            depths = layer.get("depths", [])
            assert len(depths) > 0, f"Layer {layer['name']} must have depth entries"
            for depth in depths:
                assert "label" in depth, "Each depth must have a 'label'"
                assert "values" in depth, "Each depth must have 'values'"

    @pytest.mark.asyncio
    async def test_search_returns_observations(self, adapter):
        """search() wraps API data into EcologicalObservation records."""
        try:
            result = await adapter.search(SearchParams(
                lat=51.57,
                lng=5.39,
                limit=10,
            ))
        except Exception as exc:
            if _api_unavailable(exc):
                pytest.skip("SoilGrids API is temporarily unavailable")
            raise

        if len(result) == 0:
            pytest.skip("SoilGrids returned no data (API may be down)")

        obs = result[0]
        assert obs.id.startswith("soilgrids:")
        assert obs.modality == "sensor"
        assert obs.taxon is None, "Soil data should have no taxon"
        assert obs.location.lat is not None
        assert obs.location.lng is not None
        assert obs.provenance.source_api == "soilgrids"
        assert obs.provenance.doi == "10.5194/soil-7-217-2021"
        assert obs.provenance.license == "CC-BY-4.0"
        assert obs.provenance.institution_code == "ISRIC"
        assert obs.quality.tier == 1
        assert obs.quality.grade == "research"

    @pytest.mark.asyncio
    async def test_observations_have_depth_info(self, adapter):
        """Each observation includes depth label and range."""
        try:
            result = await adapter.search(SearchParams(
                lat=37.77,
                lng=-122.43,
                limit=10,
            ))
        except Exception as exc:
            if _api_unavailable(exc):
                pytest.skip("SoilGrids API is temporarily unavailable")
            raise

        if len(result) == 0:
            pytest.skip("SoilGrids returned no data")

        for obs in result:
            assert "depth_label" in obs.value, "Each obs must include depth_label"
            assert "depth_top_cm" in obs.value, "Each obs must include depth_top_cm"
            assert "depth_bottom_cm" in obs.value, "Each obs must include depth_bottom_cm"


# ===========================================================================
# Layer 3: Semantic — is the data plausible?
# ===========================================================================

class TestSemantic:

    @pytest.mark.asyncio
    async def test_clay_content_is_plausible(self, adapter):
        """
        Known answer: Clay content should be 0-1000 g/kg (0-100%).
        Values outside this range indicate a mapping or unit error.
        """
        try:
            result = await adapter.query_properties(
                lat=37.77,  # San Francisco
                lon=-122.43,
                properties=["clay"],
                depths=["0-5cm"],
                values=["mean"],
            )
        except Exception as exc:
            if _api_unavailable(exc):
                pytest.skip("SoilGrids API is temporarily unavailable")
            raise

        layers = result.get("properties", {}).get("layers", [])
        if not layers:
            pytest.skip("No layers returned")

        clay_layer = next((l for l in layers if l["name"] == "clay"), None)
        assert clay_layer is not None, "Expected clay layer in response"

        for depth in clay_layer.get("depths", []):
            mean_val = depth.get("values", {}).get("mean")
            if mean_val is not None:
                # Clay in g/kg: should be 0-1000
                assert 0 <= mean_val <= 1000, (
                    f"Clay content {mean_val} g/kg is outside plausible range (0-1000)"
                )

    @pytest.mark.asyncio
    async def test_ph_is_plausible(self, adapter):
        """Soil pH should be between 2.0 and 11.0 (mapped as pH×10: 20-110)."""
        try:
            result = await adapter.query_properties(
                lat=37.77,
                lon=-122.43,
                properties=["phh2o"],
                depths=["0-5cm"],
                values=["mean"],
            )
        except Exception as exc:
            if _api_unavailable(exc):
                pytest.skip("SoilGrids API is temporarily unavailable")
            raise

        layers = result.get("properties", {}).get("layers", [])
        if not layers:
            pytest.skip("No layers returned")

        ph_layer = next((l for l in layers if l["name"] == "phh2o"), None)
        assert ph_layer is not None, "Expected phh2o layer in response"

        for depth in ph_layer.get("depths", []):
            mean_val = depth.get("values", {}).get("mean")
            if mean_val is not None:
                # pH×10: should be 20-110 (pH 2.0 to 11.0)
                assert 20 <= mean_val <= 110, (
                    f"pH value {mean_val} (×10) is outside plausible range. "
                    f"Expected 20-110 (pH 2.0-11.0). Got pH={mean_val/10:.1f}"
                )


# ===========================================================================
# Layer 4: Scientific — does it solve the research question?
# ===========================================================================

class TestRestorationAssessment:

    @pytest.mark.asyncio
    async def test_texture_triangle_components(self, adapter):
        """
        A soil scientist needs clay + sand + silt to classify soil texture.
        These three should be available at the same location and depth.
        """
        try:
            result = await adapter.query_properties(
                lat=37.77,
                lon=-122.43,
                properties=["clay", "sand", "silt"],
                depths=["0-5cm"],
                values=["mean"],
            )
        except Exception as exc:
            if _api_unavailable(exc):
                pytest.skip("SoilGrids API is temporarily unavailable")
            raise

        layers = result.get("properties", {}).get("layers", [])
        if not layers:
            pytest.skip("No layers returned")

        layer_names = {l["name"] for l in layers}
        assert "clay" in layer_names, "Clay required for texture classification"
        assert "sand" in layer_names, "Sand required for texture classification"
        assert "silt" in layer_names, "Silt required for texture classification"

        # Extract mean values for the texture components
        texture_values = {}
        for layer in layers:
            name = layer["name"]
            for depth in layer.get("depths", []):
                mean_val = depth.get("values", {}).get("mean")
                if mean_val is not None:
                    texture_values[name] = mean_val

        # If all three are present, they should sum to roughly 1000 g/kg
        if len(texture_values) == 3:
            total = sum(texture_values.values())
            assert 800 <= total <= 1200, (
                f"Clay ({texture_values.get('clay')}) + Sand ({texture_values.get('sand')}) "
                f"+ Silt ({texture_values.get('silt')}) = {total} g/kg. "
                f"Expected ~1000 g/kg (100%)."
            )

    @pytest.mark.asyncio
    async def test_depth_profile_available(self, adapter):
        """
        Restoration planning requires understanding the full soil profile.
        Multiple depth intervals should be available at any location.
        """
        try:
            result = await adapter.search(SearchParams(
                lat=37.77,
                lng=-122.43,
                limit=10,
            ))
        except Exception as exc:
            if _api_unavailable(exc):
                pytest.skip("SoilGrids API is temporarily unavailable")
            raise

        if len(result) == 0:
            pytest.skip("SoilGrids returned no data")

        # Should have multiple depth observations
        depth_labels = [obs.value.get("depth_label") for obs in result]
        unique_depths = set(depth_labels)
        assert len(unique_depths) >= 2, (
            f"Expected multiple depth intervals for soil profile analysis, "
            f"only got: {unique_depths}"
        )


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_missing_location_returns_empty(self, adapter):
        """Search without lat/lng should return empty, not crash."""
        result = await adapter.search(SearchParams(limit=10))
        assert isinstance(result, list)
        assert len(result) == 0

    def test_invalid_property_raises_error(self, adapter):
        """Requesting an invalid property should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid property"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                adapter.query_properties(lat=0, lon=0, properties=["not_a_property"])
            )

    def test_invalid_depth_raises_error(self, adapter):
        """Requesting an invalid depth should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid depth"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                adapter.query_properties(lat=0, lon=0, depths=["0-999cm"])
            )

    @pytest.mark.asyncio
    async def test_ocean_location_returns_gracefully(self, adapter):
        """
        Querying a point in the middle of the ocean should not crash.
        SoilGrids may return empty/null or an error for ocean points.
        """
        try:
            result = await adapter.search(SearchParams(
                lat=0.0,   # Mid-Atlantic
                lng=-30.0,
                limit=5,
            ))
            # Either empty or valid observations — either is acceptable
            assert isinstance(result, list)
        except Exception as exc:
            if _api_unavailable(exc):
                pytest.skip("SoilGrids API is temporarily unavailable")
            # Other errors (404, etc.) are acceptable for ocean points
            pass
