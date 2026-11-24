# Spec 019: Satellite Source Cards and a First Greenness Adapter

> Phase 6 — "Data must be easy to get at scale" (docs/DIRECTION.md, Data policy)
> Priority: P1
> Estimated effort: 1–2 sessions
> Dependency: Spec 015 (day-of-year climatology functions)
> Network: blocked on the runner as of 2026-11-24. Every candidate host was
> refused: modis.ornl.gov, planetarycomputer.microsoft.com,
> earth-search.aws.element84.com, appeears.earthdatacloud.nasa.gov,
> wcc.sc.egov.usda.gov.

## Objective

Satellite is the one base layer the federation lacks. Write a one-page
source card for each candidate so Christine can see licence, method,
coverage and known weaknesses side by side. Then build one adapter for live
greenness at a point, with "is this normal for here?" answered the same way
as river flow: against that pixel's own record for the time of year.

## Part 1 — Source cards: `docs/sources/<slug>.md`

One page each, same headings: **What it measures · Resolution and revisit ·
Record length · Coverage · Licence · How to get it (endpoint, auth) ·
Known weaknesses · Best first use in Kinship Earth · Sources (opened links)**.

Cards to write:

1. `modis-ndvi` — MODIS MOD13Q1 vegetation indices via the ORNL DAAC
   point-subset web service (250 m, 16-day, 2000–present).
2. `sentinel2-landsat` — Sentinel-2 L2A and Landsat C2 through a public STAC
   catalogue (10–30 m; needs raster reads).
3. `rangeland-analysis-platform` — RAP 16-day and annual forage and cover,
   western US, from 1986. Note: accuracy on California annual grassland is
   unverified (DIRECTION.md); say what validation exists.
4. `alphaearth-embeddings` — Google's AlphaEarth satellite embeddings, 10 m,
   annual 2017–2024, CC-BY 4.0; what it takes to map from a few dozen
   labelled points.
5. `snotel` — NRCS snowpack stations (snow water equivalent, daily), for
   watersheds.

Rules: every fact comes from a page that was opened (web search may find
it; the page must be fetched). If a fact cannot be sourced, write "not
confirmed" rather than guessing. Licence and attribution text quoted exactly.

## Part 2 — First adapter: point greenness from MODIS (ORNL DAAC)

Chosen first because it needs no key, returns JSON for a single point, and
has 25 years of record, which is what a day-of-year baseline needs. Revisit
this choice if the source card turns up a blocker.

**Before building, make live calls and record what you find in a comment at
the top of the adapter:** the dates endpoint, the subset endpoint and its
maximum dates per request, the scale factor and fill values for NDVI, and
the pixel-reliability (QA) band name and codes.

`servers/modis/` (new server, same layout as `servers/usgs-nwis/`):

```python
class MODISAdapter(EcologicalAdapter):
    async def get_ndvi_record(self, lat: float, lon: float,
                              start: date | None = None, end: date | None = None
                              ) -> list[tuple[date, float]]
        # 16-day composites, oldest first; drops fill values and pixels whose
        # reliability flag is not "good" or "marginal"; disk cache 16 days.
```

Then in `shared/src/kinship_shared/baselines.py`, reuse the spec 015
climatology for greenness:

```python
def greenness_percentile(value, record, target, window_days=16, min_years=10) -> float | None
```

and an MCP tool `ecology_greenness(lat, lon)` returning the latest
composite, its date, its percentile for the time of year, the record
length, and a plain sentence ("Greener than 85% of late-November
composites here since 2000").

## Tests

1. Offline: QA filtering drops bad and fill values from a fixture response.
2. Offline: `greenness_percentile` matches `flow_percentile` behaviour on
   the same synthetic record (shared code path).
3. Offline: tool output names the date, percentile and record length.
4. Network (`@pytest.mark.network`): a Russian River valley point returns ≥ 20 years of composites.

## Do Not

- Do not add a raster stack (rasterio, GDAL) in this spec; point services only.
- Do not claim forage or biomass from NDVI; greenness only.
- Do not write card facts from memory.

## Hosts to allow before building

`modis.ornl.gov` (adapter); for the cards, the documentation sites of each
source (`modis.ornl.gov`, `planetarycomputer.microsoft.com` or
`earth-search.aws.element84.com`, `rangelands.app`,
`developers.google.com`, `www.nrcs.usda.gov`, `wcc.sc.egov.usda.gov`).
Confirm exact hostnames when writing the cards.

## Commit

```
Satellite source cards and MODIS greenness adapter (spec 019)

- docs/sources/: 5 source cards with opened sources
- servers/modis: point NDVI record with QA filtering and cache
- greenness_percentile + ecology_greenness tool
- 4 tests
```
