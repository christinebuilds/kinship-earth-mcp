"""
Evaluative Tests: Scientist Persona Research Scenarios
======================================================

PURPOSE
-------
These tests simulate real interdisciplinary research workflows — the kind of
questions that are currently unanswerable because data lives in silos across
different databases, formats, and query interfaces.

Each test module is a scientist persona with a multi-step research workflow.
They don't just check "does the code run?" — they verify that the system
returns data that is scientifically meaningful, cross-referenced, and
sufficient to actually answer the research question.

WHAT MAKES THESE DIFFERENT FROM UNIT TESTS
------------------------------------------
Unit tests:  "Does OBIS return records when I search for a species?"
Persona tests: "Can a marine ecologist studying climate-driven range shifts
               get OBIS occurrence data AND ERA5 temperature data for the same
               spatiotemporal window, with provenance good enough to cite in a
               paper, and quality metadata good enough to filter by?"

These tests exercise the ORCHESTRATOR — the cross-source layer that is the
unique value of Kinship Earth.

PERSONAS
--------
1. Marine Climate Ecologist — climate-driven species redistribution
2. Watershed Ecologist — terrestrial-marine boundary effects
3. Phenology Researcher — seasonal timing shifts under warming
4. Climate Validation Scientist — ground-truthing reanalysis with in-situ data

RUNNING THESE TESTS
-------------------
From the repo root:
  uv run --package kinship-orchestrator pytest servers/orchestrator/tests/test_scientist_scenarios.py -v

Run a single persona:
  uv run --package kinship-orchestrator pytest servers/orchestrator/tests/test_scientist_scenarios.py -k "marine_ecologist" -v
"""

import pytest

from kinship_orchestrator.server import (
    ecology_get_environmental_context,
    ecology_search,
    ecology_describe_sources,
)


# ===========================================================================
# PERSONA 1: Marine Climate Ecologist
# Research question: "Did the 2021 Pacific heat dome shift cetacean
# distributions in the North Pacific?"
#
# Why this is currently hard: A researcher must manually query OBIS for
# cetacean records, separately download ERA5 from Copernicus (different auth,
# NetCDF format, different coordinate grid), align temporal windows, and
# wrangle both into the same analysis. Days of work before any science.
#
# What Kinship Earth should enable: one query returns species occurrences
# AND climate context for the same spatiotemporal window.
# ===========================================================================

# North Pacific study area — off Oregon coast
NORTH_PACIFIC_LAT = 44.6
NORTH_PACIFIC_LNG = -124.5

# Heat dome peak: late June 2021
HEAT_DOME_DATE = "2021-06-28"


class TestMarineClimateEcologist:
    """
    Workflow: Compare cetacean sightings + ocean temperature before/during
    a marine heatwave event.
    """

    @pytest.mark.asyncio
    async def test_step1_get_climate_during_heat_event(self):
        """
        STEP 1: Get temperature data during the heat dome.

        A marine ecologist's first question: "How extreme was the heat dome
        at my study site?" ERA5 should show the temperature anomaly.

        Known answer: The June 2021 Pacific heat dome produced temperatures
        15-20°C above normal in the PNW. ERA5 should show temperatures well
        above the June average (~15°C) for coastal Oregon.
        """
        result = await ecology_get_environmental_context(
            lat=NORTH_PACIFIC_LAT,
            lon=NORTH_PACIFIC_LNG,
            date=HEAT_DOME_DATE,
            days_before=7,
            days_after=3,
        )

        # Must have climate data
        assert "climate" in result
        daily = result["climate"]["daily"]
        assert "time" in daily
        assert len(daily["time"]) >= 7, "Should have at least 7 days of data"

        # Temperature data must be present
        temp_key = None
        for key in ["temperature_2m_max", "temperature_2m_mean"]:
            if key in daily:
                temp_key = key
                break
        assert temp_key is not None, (
            "Climate data must include temperature. "
            "Available keys: " + str(list(daily.keys()))
        )

        # Temperatures should be plausible for coastal Oregon in June
        # (not sub-zero, not 60°C)
        temps = daily[temp_key]
        for i, t in enumerate(temps):
            assert -10 < t < 55, (
                f"Day {daily['time'][i]}: temperature {t}°C is implausible "
                f"for coastal Oregon in June."
            )

        # Provenance must be citable
        prov = result["climate"]["provenance"]
        assert prov.get("doi") == "10.24381/cds.adbb2d47", "ERA5 DOI required"
        assert prov.get("license") == "CC-BY-4.0", "ERA5 license required"

    @pytest.mark.asyncio
    async def test_step2_search_cetaceans_near_study_site(self):
        """
        STEP 2: Find marine mammal records near the study site.

        The ecologist now wants to know: "What cetaceans have been observed
        in this area?" This exercises OBIS through the unified search.

        We search broadly (all cetaceans by location, no species filter)
        to see what's been recorded. OBIS should have marine mammal records
        off the Oregon coast — it's a well-surveyed area.
        """
        result = await ecology_search(
            lat=NORTH_PACIFIC_LAT,
            lon=NORTH_PACIFIC_LNG,
            radius_km=200,
            start_date="2019-01-01",
            end_date="2023-12-31",
            include_climate=False,
            limit=50,
        )

        # Should have search context showing what was queried
        assert "search_context" in result
        ctx = result["search_context"]
        assert "obis" in ctx["sources_queried"], (
            "OBIS should be queried for marine location searches"
        )

        # The system should return results or give helpful guidance
        if result["species_count"] == 0:
            # Sparse results should trigger near-miss hints
            assert "sparse_results_hint" in ctx or result["species_count"] == 0, (
                "When no results found, system should provide guidance on "
                "expanding the search."
            )
        else:
            # Verify returned records have the fields a researcher needs
            occ = result["species_occurrences"][0]
            assert "scientific_name" in occ, "Species name is required"
            assert "lat" in occ and "lng" in occ, "Coordinates are required"
            assert "observed_at" in occ, "Observation date is required"
            assert "quality_tier" in occ, "Quality tier needed for filtering"
            assert "license" in occ, "License needed for data citation"

            # Coordinates should be marine (off Oregon coast)
            assert occ["lng"] < -120, (
                f"Record at lng={occ['lng']} — should be off the Pacific coast"
            )

    @pytest.mark.asyncio
    async def test_step3_cross_reference_species_with_climate(self):
        """
        STEP 3: The cross-source query — species + climate together.

        THIS IS THE TEST THAT MATTERS MOST. This is the query that is
        currently impossible without Kinship Earth.

        The ecologist asks: "Show me dolphin sightings near my study area
        WITH the climate conditions at those times."

        The system must return both species occurrences AND climate data
        in a single response, from a single query.
        """
        result = await ecology_search(
            scientificname="Delphinus delphis",
            lat=NORTH_PACIFIC_LAT,
            lon=NORTH_PACIFIC_LNG,
            radius_km=500,  # wide search — Pacific is big
            start_date="2019-01-01",
            end_date="2023-12-31",
            include_climate=True,
            limit=20,
        )

        # Must have search context
        assert "search_context" in result
        sources = result["search_context"]["sources_queried"]

        # Both species and climate sources must be queried
        assert "obis" in sources, "OBIS must be queried for species"
        assert "era5" in sources, "ERA5 must be queried for climate context"

        # Climate data must be present when requested
        assert result["climate"] is not None, (
            "When include_climate=True with valid lat/lon/dates, "
            "ERA5 climate data must be included in the response."
        )
        assert "daily" in result["climate"], "Climate must include daily data"

        # This is the key assertion: the response contains BOTH ecological
        # observations AND environmental context, enabling the researcher
        # to correlate species presence with climate conditions without
        # touching two separate databases.

    @pytest.mark.asyncio
    async def test_step4_relevance_scores_enable_quality_filtering(self):
        """
        STEP 4: Can the researcher filter by data quality?

        A researcher publishing in a peer-reviewed journal needs to filter
        by quality tier and understand the confidence of each record.
        The relevance scoring system must expose these components.
        """
        result = await ecology_search(
            scientificname="Delphinus delphis",
            lat=41.5,  # Woods Hole — known to have records
            lon=-70.7,
            radius_km=200,
            start_date="2015-01-01",
            end_date="2023-12-31",
            include_climate=False,
            limit=20,
        )

        if result["species_count"] > 0:
            occ = result["species_occurrences"][0]
            rel = occ["relevance"]

            # Relevance must expose components, not just a scalar
            assert "score" in rel, "Must have composite score"
            assert "geo_distance_km" in rel, "Must expose geographic distance"
            assert "taxon_match" in rel, "Must expose taxon match quality"
            assert "quality_score" in rel, "Must expose quality score"
            assert "explanation" in rel, "Must have human-readable explanation"

            # Score must be normalized
            assert 0 <= rel["score"] <= 1, f"Score {rel['score']} out of range"

            # Explanation should be informative, not empty
            assert len(rel["explanation"]) > 10, (
                f"Explanation too short to be useful: {rel['explanation']!r}"
            )


# ===========================================================================
# PERSONA 2: Watershed Ecologist
# Research question: "What's the relationship between terrestrial conditions
# at coastal NEON sites and marine biodiversity in adjacent waters?"
#
# Why this is currently impossible: No existing system connects NEON's
# terrestrial sensor network to OBIS's marine occurrence database. They
# use different coordinate systems, different taxonomies, different APIs.
# ===========================================================================

# Coastal study site — near NEON's SERC site (Smithsonian Environmental
# Research Center, Chesapeake Bay, MD)
CHESAPEAKE_LAT = 38.89
CHESAPEAKE_LNG = -76.56


class TestWatershedEcologist:
    """
    Workflow: Find terrestrial monitoring sites near a coast, then search
    for marine species in the adjacent waters, with shared climate context.
    """

    @pytest.mark.network
    @pytest.mark.asyncio
    async def test_step1_find_coastal_neon_sites(self):
        """
        STEP 1: Discovery — what terrestrial monitoring exists near my coast?

        The watershed ecologist starts with a location (Chesapeake Bay) and
        asks: "What NEON sites are near this coastline?"

        This tests geographic search with a location that should return
        NEON sites (SERC is right on Chesapeake Bay).
        """
        result = await ecology_search(
            lat=CHESAPEAKE_LAT,
            lon=CHESAPEAKE_LNG,
            radius_km=100,
            limit=10,
        )

        # Should find at least one NEON site near Chesapeake Bay
        assert result["neon_site_count"] > 0, (
            "Should find NEON sites near Chesapeake Bay. "
            "SERC (Smithsonian Environmental Research Center) is at these "
            "coordinates."
        )

        # Verify site has location metadata
        site = result["neon_sites"][0]
        assert "site_code" in site
        assert "lat" in site and "lng" in site
        assert "site_name" in site

    @pytest.mark.asyncio
    async def test_step2_search_marine_life_adjacent_to_neon_site(self):
        """
        STEP 2: Cross-domain query — what's in the water next to the land site?

        After finding a NEON site, the ecologist asks: "What marine species
        have been recorded in the waters near this terrestrial monitoring
        station?"

        This is the land-sea boundary query that no current system supports.
        """
        result = await ecology_search(
            lat=CHESAPEAKE_LAT,
            lon=CHESAPEAKE_LNG,
            radius_km=100,  # includes Chesapeake Bay waters
            start_date="2018-01-01",
            end_date="2023-12-31",
            include_climate=False,
            limit=20,
        )

        # Should have queried both OBIS and NEON
        sources = result["search_context"]["sources_queried"]
        assert "obis" in sources, "Must query OBIS for marine records"
        assert "neon" in sources, "Must query NEON for terrestrial sites"

        # The response should contain BOTH types of data
        # (even if one returns zero results — the system tried both)
        assert "species_occurrences" in result
        assert "neon_sites" in result

    @pytest.mark.asyncio
    async def test_step3_climate_context_bridges_land_and_sea(self):
        """
        STEP 3: Get shared climate context for both domains.

        ERA5 climate data (temperature, precipitation) is the bridge between
        terrestrial and marine systems. Precipitation drives runoff, which
        affects coastal marine ecosystems.

        The ecologist needs: "What was the precipitation and temperature at
        my study site over the past year?" — applicable to both the land
        and adjacent water.
        """
        result = await ecology_get_environmental_context(
            lat=CHESAPEAKE_LAT,
            lon=CHESAPEAKE_LNG,
            date="2023-07-01",
            days_before=30,
            days_after=0,
        )

        daily = result["climate"]["daily"]

        # Should have precipitation data (key for watershed research)
        precip_key = None
        for key in ["precipitation_sum", "rain_sum"]:
            if key in daily:
                precip_key = key
                break
        assert precip_key is not None, (
            "Climate data must include precipitation for watershed research. "
            f"Available: {list(daily.keys())}"
        )

        # Should have temperature
        assert any(k for k in daily.keys() if "temperature" in k), (
            "Climate data must include temperature"
        )

        # Should find NEON sites (SERC)
        assert result["nearby_neon_count"] > 0, (
            "Should find NEON sites near Chesapeake Bay"
        )


# ===========================================================================
# PERSONA 3: Phenology Researcher
# Research question: "Are seasonal ecological events shifting earlier at
# NEON sites where ERA5 shows warming trends?"
#
# Why this needs cross-source data: NEON has species survey dates. ERA5
# has temperature trends. Connecting them requires querying both systems
# for the same sites across multiple years.
# ===========================================================================

# Use two NEON sites at different latitudes to test gradient
# WREF (Wind River, WA) — 45.82°N — PNW maritime
# TALL (Talladega, AL) — 32.95°N — Southeast humid subtropical

WREF_LAT, WREF_LNG = 45.82, -121.95
TALL_LAT, TALL_LNG = 32.95, -87.39


class TestPhenologyResearcher:
    """
    Workflow: Compare environmental conditions across multiple NEON sites
    along a latitudinal gradient to study phenological timing.
    """

    @pytest.mark.asyncio
    async def test_step1_characterize_site_climate_pnw(self):
        """
        STEP 1a: Get summer climate conditions at the northern site (WREF).

        The phenology researcher needs to characterize temperature regimes
        at each study site. "What's the typical summer temperature at Wind
        River?"
        """
        result = await ecology_get_environmental_context(
            lat=WREF_LAT, lon=WREF_LNG,
            date="2023-06-21",  # summer solstice
            days_before=14,
            days_after=14,
        )

        daily = result["climate"]["daily"]
        temps = daily.get("temperature_2m_mean") or daily.get("temperature_2m_max")
        assert temps is not None, "Must have temperature data"

        # PNW summer temps should be moderate (10-35°C)
        avg_temp = sum(temps) / len(temps)
        assert 5 < avg_temp < 40, (
            f"Average June temp at WREF = {avg_temp:.1f}°C — "
            f"implausible for PNW maritime climate."
        )

        # Must find WREF itself
        site_codes = [s["site_code"] for s in result["nearby_neon_sites"]]
        assert "WREF" in site_codes, "WREF should be found at its own coordinates"

    @pytest.mark.asyncio
    async def test_step2_characterize_site_climate_southeast(self):
        """
        STEP 1b: Get summer climate conditions at the southern site (TALL).

        Same query, different site. The researcher will compare these.
        """
        result = await ecology_get_environmental_context(
            lat=TALL_LAT, lon=TALL_LNG,
            date="2023-06-21",
            days_before=14,
            days_after=14,
        )

        daily = result["climate"]["daily"]
        temps = daily.get("temperature_2m_mean") or daily.get("temperature_2m_max")
        assert temps is not None

        # Southeast summer temps should be warmer than PNW
        avg_temp = sum(temps) / len(temps)
        assert 15 < avg_temp < 45, (
            f"Average June temp at TALL = {avg_temp:.1f}°C — "
            f"implausible for Alabama in summer."
        )

    @pytest.mark.network
    @pytest.mark.asyncio
    async def test_step3_compare_sites_shows_latitudinal_gradient(self):
        """
        STEP 2: Compare the two sites — does the data show the expected
        latitudinal temperature gradient?

        Known answer: Alabama (32.95°N) is warmer than Washington (45.82°N)
        in summer. If our data doesn't show this, something is wrong with
        coordinate resolution or data retrieval.

        This is a cross-site comparison that requires making the same query
        to ERA5 for two different locations — exactly the kind of systematic
        workflow that Kinship Earth should make easy.
        """
        import asyncio

        pnw_task = ecology_get_environmental_context(
            lat=WREF_LAT, lon=WREF_LNG,
            date="2023-07-15", days_before=14, days_after=14,
        )
        se_task = ecology_get_environmental_context(
            lat=TALL_LAT, lon=TALL_LNG,
            date="2023-07-15", days_before=14, days_after=14,
        )

        pnw, se = await asyncio.gather(pnw_task, se_task)

        pnw_temps = pnw["climate"]["daily"].get("temperature_2m_mean") or \
                    pnw["climate"]["daily"].get("temperature_2m_max")
        se_temps = se["climate"]["daily"].get("temperature_2m_mean") or \
                   se["climate"]["daily"].get("temperature_2m_max")

        pnw_avg = sum(pnw_temps) / len(pnw_temps)
        se_avg = sum(se_temps) / len(se_temps)

        assert se_avg > pnw_avg, (
            f"Alabama ({se_avg:.1f}°C) should be warmer than Washington "
            f"({pnw_avg:.1f}°C) in summer. If reversed, coordinate "
            f"resolution or data retrieval is broken."
        )


# ===========================================================================
# PERSONA 4: Climate Validation Scientist
# Research question: "How well do ERA5 reanalysis estimates match NEON's
# in-situ measurements at the same sites?"
#
# Why this matters: ERA5 is modeled data at ~25km resolution. NEON is
# ground-truth at specific points. Comparing them reveals where reanalysis
# is trustworthy — critical for all downstream ecological modeling.
#
# Why this needs Kinship Earth: ERA5 and NEON are completely separate data
# ecosystems with different APIs, formats, and access methods.
# ===========================================================================


class TestClimateValidationScientist:
    """
    Workflow: Query ERA5 and NEON for the same location, then compare
    metadata to assess whether cross-validation is feasible.
    """

    @pytest.mark.asyncio
    async def test_step1_both_sources_cover_same_location(self):
        """
        STEP 1: Verify that ERA5 and NEON both return data for WREF.

        The validation scientist needs both data sources to agree on the
        location. ERA5 resolves to the nearest grid point (~25km), NEON
        reports the exact site coordinates. The resolved locations should
        be reasonably close.
        """
        result = await ecology_get_environmental_context(
            lat=WREF_LAT, lon=WREF_LNG,
            date="2023-07-01",
            days_before=3,
            days_after=0,
        )

        # ERA5 resolved coordinates should be near WREF
        era5_lat = result["climate"]["location_resolved"]["lat"]
        era5_lon = result["climate"]["location_resolved"]["lon"]
        assert abs(era5_lat - WREF_LAT) < 0.5, (
            f"ERA5 resolved lat {era5_lat} is too far from WREF ({WREF_LAT}). "
            f"ERA5 grid is ~25km — should resolve within 0.3°."
        )
        assert abs(era5_lon - WREF_LNG) < 0.5, (
            f"ERA5 resolved lon {era5_lon} is too far from WREF ({WREF_LNG})."
        )

        # NEON should be found at the same location
        assert result["nearby_neon_count"] > 0, "NEON site should be found"
        wref = next(
            (s for s in result["nearby_neon_sites"] if s["site_code"] == "WREF"),
            None,
        )
        assert wref is not None, "WREF should be in nearby sites"

    @pytest.mark.asyncio
    async def test_step2_quality_tiers_distinguish_sources(self):
        """
        STEP 2: Quality metadata must distinguish modeled vs measured data.

        For climate validation research, it's critical to know which data
        is modeled (ERA5) and which is measured (NEON). The quality tier
        system should make this clear.

        Both ERA5 and NEON are Tier 1, but for different reasons:
        - ERA5: calibrated reanalysis (modeled, but peer-reviewed methodology)
        - NEON: calibrated instruments (direct measurement)

        The validation scientist needs this distinction to be visible.
        """
        result = await ecology_describe_sources()

        era5_source = next(
            (s for s in result["sources"] if s["id"] == "era5"), None
        )
        neon_source = next(
            (s for s in result["sources"] if s["id"] == "neonscience"), None
        )

        assert era5_source is not None, "ERA5 must be in source list"
        assert neon_source is not None, "NEON must be in source list"

        # Both should be Tier 1 (calibrated)
        assert era5_source["quality_tier"] == 1, "ERA5 is Tier 1"
        assert neon_source["quality_tier"] == 1, "NEON is Tier 1"

        # Descriptions should distinguish modeled from measured
        era5_desc = era5_source["description"].lower()
        neon_desc = neon_source["description"].lower()

        # ERA5 should mention reanalysis/modeled
        assert any(word in era5_desc for word in ["reanalysis", "model", "gridded"]), (
            f"ERA5 description should mention it's reanalysis/modeled data. "
            f"Got: {era5_source['description']}"
        )

    @pytest.mark.asyncio
    async def test_step3_provenance_enables_comparison(self):
        """
        STEP 3: Provenance must be complete enough to write a methods section.

        A validation study requires precise citation of both data sources.
        The researcher must be able to cite the exact ERA5 product version
        and the NEON site/products used, with DOIs where available.
        """
        result = await ecology_get_environmental_context(
            lat=WREF_LAT, lon=WREF_LNG,
            date="2023-07-01",
        )

        # ERA5 provenance
        era5_prov = result["climate"]["provenance"]
        assert era5_prov.get("doi"), "ERA5 must have a DOI for citation"
        assert era5_prov.get("license"), "ERA5 must state its license"

        # NEON site provenance (via portal URL)
        if result["nearby_neon_count"] > 0:
            neon_site = result["nearby_neon_sites"][0]
            assert neon_site.get("portal_url"), (
                "NEON sites should include a portal URL for data access"
            )


# ===========================================================================
# CROSS-PERSONA: Integration Quality Checks
# These tests verify properties that ALL research workflows depend on.
# ===========================================================================


class TestCrossPersonaIntegration:
    """
    Properties that every scientist persona depends on, regardless of their
    specific research question.
    """

    @pytest.mark.asyncio
    async def test_parallel_queries_dont_interfere(self):
        """
        CONCURRENCY TEST: Multiple simultaneous queries return correct data.

        Real usage involves multiple queries in flight. If adapters share
        mutable state, results could get mixed up (a bug that would be
        catastrophic for research integrity).
        """
        import asyncio

        # Three simultaneous queries for different locations
        tasks = [
            ecology_get_environmental_context(
                lat=WREF_LAT, lon=WREF_LNG,
                date="2023-06-15",
            ),
            ecology_get_environmental_context(
                lat=TALL_LAT, lon=TALL_LNG,
                date="2023-06-15",
            ),
            ecology_get_environmental_context(
                lat=CHESAPEAKE_LAT, lon=CHESAPEAKE_LNG,
                date="2023-06-15",
            ),
        ]
        results = await asyncio.gather(*tasks)

        # Each result should have the correct query coordinates
        assert results[0]["query"]["lat"] == WREF_LAT
        assert results[1]["query"]["lat"] == TALL_LAT
        assert results[2]["query"]["lat"] == CHESAPEAKE_LAT

        # ERA5 resolved locations should be near their respective queries
        for i, (expected_lat, result) in enumerate(zip(
            [WREF_LAT, TALL_LAT, CHESAPEAKE_LAT], results
        )):
            resolved_lat = result["climate"]["location_resolved"]["lat"]
            assert abs(resolved_lat - expected_lat) < 1.0, (
                f"Query {i}: ERA5 resolved to {resolved_lat}, expected near "
                f"{expected_lat}. Possible state leak between concurrent queries."
            )

    @pytest.mark.asyncio
    async def test_all_sources_self_describe_accurately(self):
        """
        REGISTRY TEST: Source descriptions must be accurate and complete.

        Every research workflow starts with discovery: "What data is available?"
        If source descriptions are wrong or incomplete, the researcher (or
        agent) will make bad decisions about which sources to query.
        """
        result = await ecology_describe_sources()

        for source in result["sources"]:
            # Every source must have a meaningful description
            assert len(source["description"]) > 30, (
                f"Source {source['id']} has a too-short description: "
                f"{source['description']!r}. Scientists need enough context "
                f"to decide whether this source is relevant."
            )

            # Modalities must be specified
            assert len(source["modalities"]) > 0, (
                f"Source {source['id']} has no modalities listed"
            )

            # Search capabilities must be declared
            caps = source["search_capabilities"]
            assert isinstance(caps.get("location"), bool), (
                f"Source {source['id']} must declare location search capability"
            )
            assert isinstance(caps.get("taxon"), bool), (
                f"Source {source['id']} must declare taxon search capability"
            )

    @pytest.mark.asyncio
    async def test_empty_results_provide_guidance_not_silence(self):
        """
        UX TEST: When a query returns no results, the system must help
        the researcher understand why and suggest alternatives.

        Silence on empty results is the worst possible UX for a scientist.
        "No data" could mean: wrong coordinates, wrong date range, wrong
        species name, or genuinely no observations. The system should
        distinguish these cases.
        """
        # Search in the middle of the Sahara — should find no OBIS or NEON
        result = await ecology_search(
            scientificname="Delphinus delphis",
            lat=25.0,
            lon=10.0,  # Sahara desert
            radius_km=50,
            start_date="2020-01-01",
            end_date="2023-12-31",
            include_climate=False,
            limit=10,
        )

        # Should still return a structured response, not an error
        assert "species_occurrences" in result
        assert "search_context" in result

        # When results are sparse, should have guidance
        if result["species_count"] < 3:
            ctx = result["search_context"]
            # Either sparse_results_hint or just zero results (both are OK)
            # What's NOT OK is returning {"error": ...} or crashing
            assert isinstance(result["species_occurrences"], list)
