# Spec 023: Neighborhood Flood Outlook — "Where will it flood if we get 1–2 inches this week?"

> Phase 6 — from issue #18 (`data-gap`), proposed 2026-11-23
> Priority: P1 (pending Christine's decision to queue it)
> Estimated effort: 2 sessions (tool + scorecard)
> Dependency: Spec 016's replay/scoring module (`shared/src/kinship_shared/replay.py`)

## Objective

One tool that answers a city or resident asking, ahead of a storm: **for this
spot, when is rain heaviest this week, does that overlap high tide, and is the
spot in a known flood-prone area?** Then a published record of how often that
rule matched real street-flooding reports. San Francisco first, because its
flood-risk layer and 311 reports are open; elsewhere the tool says plainly
which piece it lacks.

This is "is this normal for here, and what usually follows?" asked of rain,
and it serves a named user directly: cities planning for El Niño winters.

## Why rain intensity, not totals

Street flooding follows how much falls in an hour or three, not the weekly
sum. An inch over five days rarely floods a street; half an inch in an hour
at high tide often does. The tool reports totals for context, but the rule
works on short-window intensity.

## Data (all open, no keys) — probe each live before building

Network access on the runner environment (2026-11-23): only
`archive-api.open-meteo.com` answers. Hosts to allow before building:
`api.open-meteo.com`, `api.tidesandcurrents.noaa.gov`, `data.sfgov.org`, and
optionally `api.weather.gov`.

| Need | Source | Host | Verify live before building |
|---|---|---|---|
| Hourly rain forecast, 7 days | Open-Meteo forecast (`hourly=precipitation`) | api.open-meteo.com | Units (mm), hours returned, model used |
| Hourly rain history (for scoring) | Open-Meteo archive, already used by spec 015 | archive-api.open-meteo.com | Reachable today |
| Second forecast opinion (optional) | NWS gridpoint `quantitativePrecipitation` (6-hour) | api.weather.gov | Needs a User-Agent; 6-hour windows only |
| Tide predictions | NOAA CO-OPS `datagetter`, `product=predictions`, `interval=hilo` and hourly; station San Francisco 9414290 | api.tidesandcurrents.noaa.gov | Nearest-station lookup via CO-OPS metadata API; datum MLLW |
| Flood-prone area | SFPUC 100-Year Storm Flood Risk Zone, DataSF id `jzu3-4yxp` (from issue #18) | data.sfgov.org | Geometry field name and format (GeoJSON via `.geojson`) |
| Past flooding (reference events) | SF 311 cases, category for flooding / sewer backup | data.sfgov.org | Dataset id and exact category names: **look them up, do not assume** |

Record what each probe finds in a comment at the top of the adapter code, as
spec 015 did for USGS paging.

## What to Build

### 1. Small adapters — `shared/src/kinship_shared/flood_sources.py`

```python
async def hourly_rain_forecast(lat, lon, days=7) -> list[tuple[datetime, float]]   # mm/h, UTC
async def hourly_rain_history(lat, lon, start: date, end: date) -> list[tuple[datetime, float]]
async def tide_predictions(lat, lon, start: date, end: date) -> TideSeries | None   # None if no station within 25 km
async def in_flood_zone(lat, lon) -> FloodZoneResult   # {known_layer: bool, in_zone: bool | None, layer, layer_updated}
```

- Flood-zone polygons cached on disk (`~/.kinship-earth/cache/datasf/jzu3-4yxp.geojson`, 30-day reuse); point-in-polygon with the stdlib (ray casting), no shapely.
- Outside San Francisco: `known_layer=False`, `in_zone=None`. Never guess.

### 2. Pure logic — `shared/src/kinship_shared/flood_outlook.py`

```python
class RainWindow(BaseModel):
    start: datetime; end: datetime; total_mm: float; max_1h_mm: float; max_3h_mm: float
    high_tide_overlap: bool | None      # None when no tide station

class FloodOutlook(BaseModel):
    lat: float; lon: float; days: int
    total_mm: float
    windows: list[RainWindow]           # 6-hour windows
    peak: RainWindow | None
    flood_zone: FloodZoneResult
    rule_fires: bool | None             # None when a needed input is missing
    rule: str                           # the rule in one plain sentence, with thresholds
    sources: list[dict]                 # name, url, fetched_at
    notes: list[str]

def build_outlook(rain, tides, zone, intensity_mm_3h: float = 12.7) -> FloodOutlook
```

- High-tide overlap: any hour of the window within ±2 h of a predicted high
  tide above the station's mean higher high water (MHHW).
- Default rule (stated, not tuned): **fires when the point is in the flood
  zone and some 3-hour window brings ≥ 12.7 mm (0.5 in)**; the note says
  whether that window overlaps high tide. The threshold is a starting point
  to be scored, not a claim.

### 3. MCP tool — `ecology_flood_outlook(lat, lon, days=7)`

Returns `FloodOutlook` as JSON plus a two-sentence plain summary, e.g.
"Heaviest rain Thursday 2–5 pm, about 0.6 in, during high tide. This block is
in the city's 100-year storm flood zone, so street flooding is possible."
Each source carries its fetch time.

### 4. Scorecard — `tools/flood_scorecard.py`

Replay 2015–present at SF: for each 3-hour window in the hourly rain history,
does the rule fire for flood-zone points? Reference events: 311 flooding
cases, **aggregated to flood-zone polygon and day** (never commit addresses
or case text). Score with spec 016's approach: hit = a reference day with the
rule firing within 12 h before; report unmatched firings as "unmatched",
not false alarms. Sweep the 3-hour threshold at 6.4, 12.7 and 19 mm and
report all three. Outputs `canon/scorecards/flood-outlook.{json,md}`.

## Tests (offline, fixtures)

1. 6-hour windowing and 1 h / 3 h maxima on a synthetic hourly series.
2. High-tide overlap true and false cases with a synthetic tide series.
3. Point-in-polygon on a small hand-made polygon, including a hole.
4. Outside SF: `known_layer=False`, `rule_fires=None`, and a note saying why.
5. Rule fires only when both zone and intensity conditions hold.
6. 311 aggregation drops addresses and keeps (zone, day) counts only.
7. Network test (`@pytest.mark.network`): live forecast returns ≥ 7×24 hours.

## Do Not

- Do not tune the threshold to the best hit rate; report the sweep.
- Do not present the outlook as an official warning. Every answer says
  "not an official flood warning; check NWS and your city".
- Do not commit 311 addresses, case text, or anything finer than
  flood-zone polygon × day.
- Do not guess flood zones outside cities with an open layer.

## Definition of Done

- `ecology_flood_outlook` answers for a point in San Francisco with rain
  windows, tide overlap and flood-zone status, each with its source.
- A flood-outlook scorecard is committed with the threshold sweep.
- Issue #18 is closed with a link to the tool and the scorecard.
