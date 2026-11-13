# Spec 022: Neighborhood Companion — "What changed near home?"

> Phase 6 — alongside the river work (Christine's decision, 2026-11-13)
> Priority: P1
> Estimated effort: 1 session (small)
> Dependency: None. Buildable and testable offline.

## Objective

One new tool that answers the daily question a curious person asks about the
place they live: **what is new here, what has left, and is that on time?**
`ecology_whats_around_me` already lists recent sightings. It does not say what
*changed*. This spec adds the comparison, and nothing else.

It is the same product question as the river work ("is this normal for
here?"), asked of open sightings instead of stream gauges.

## What to Build

### 1. Pure comparison — `shared/src/kinship_shared/seasonal_changes.py`

```python
class SpeciesWindow(BaseModel):
    """What was seen in one place over one stretch of days."""
    start: date
    end: date
    species: dict[str, SpeciesTally]      # key: scientific name

class SpeciesTally(BaseModel):
    scientific_name: str
    common_name: str | None = None
    taxon_group: str | None = None        # "bird", "plant", "insect", ... when the source gives it
    n_observations: int
    first_seen: date
    last_seen: date
    sources: list[str]                    # adapter ids

class SeasonalChange(BaseModel):
    scientific_name: str
    common_name: str | None
    taxon_group: str | None
    kind: Literal["arrived", "departed", "more_often", "less_often"]
    n_now: int
    n_before: int
    first_seen: date | None               # for "arrived"
    last_seen: date | None                # for "departed"
    last_year: Literal["also_present", "absent", "unknown"]
    timing_vs_last_year_days: int | None  # arrived only: negative = earlier this year
    note: str                             # one plain sentence

def tally(observations: list[EcologicalObservation], start: date, end: date) -> SpeciesWindow
def seasonal_changes(now: SpeciesWindow, before: SpeciesWindow,
                     last_year: SpeciesWindow | None = None,
                     min_observations: int = 2) -> list[SeasonalChange]
```

Rules:

- `arrived`: in `now` with at least `min_observations`, absent from `before`.
- `departed`: in `before` with at least `min_observations`, absent from `now`.
- `more_often` / `less_often`: present in both, count changed by a factor of
  3 or more **and** by at least 5 observations. Counts are sightings, not
  animals; the note must say "reported more often", never "more abundant".
- `last_year`: whether the species was in last year's matching window.
  `unknown` when `last_year` is `None`.
- `timing_vs_last_year_days`: for `arrived` species also present last year,
  this year's `first_seen` day-of-year minus last year's.
- Sort: arrived, departed, more_often, less_often; within each, most
  observations first.
- A single sighting never produces a change (that is what `min_observations`
  is for). Rare one-off records are noise at neighborhood scale.

### 2. Tool — `ecology_whats_changed` in `servers/orchestrator/.../server.py`

```python
async def ecology_whats_changed(lat: float, lon: float, radius_km: float = 5,
                                window_days: int = 14) -> dict
```

- Three windows: `now` = last `window_days`; `before` = the `window_days`
  before that; `last_year` = the `now` window shifted back one year, widened
  by 7 days each side.
- Sources: iNaturalist (no key) and eBird (when `EBIRD_API_KEY` is set). Call
  the adapters' `search` directly with `limit=200` per window; do not go
  through `run_search` (its limit of 30 is too small to compare windows).
- eBird's recent-observations API reaches back 30 days only. If the eBird
  adapter cannot serve the `last_year` window, use iNaturalist alone for that
  window and say so in the result (`last_year_sources`).
- Return:

```python
{
  "location": {"lat": ..., "lon": ..., "radius_km": ...},
  "windows": {"now": [start, end], "before": [...], "last_year": [...]},
  "observations": {"now": n, "before": n, "last_year": n},
  "changes": [SeasonalChange as dict, ...],      # at most 25
  "summary": "3 arrivals, 1 departure in the last 14 days.",
  "caveat": "...",
  "sources_queried": [...], "last_year_sources": [...],
}
```

- `caveat` is always present and always says: these are reports by
  volunteers, so a change can reflect who went out looking as much as what
  was there. If `observations.now` or `observations.before` is under 20, add
  that the sample is small.
- Coordinates in the stored conversation turn are rounded to 2 decimals
  (about 1 km). A home location is private.
- Do not return raw eBird records; return tallies only (eBird terms, see
  docs/DIRECTION.md, Data policy).

### 3. Docs

- README: add `ecology_whats_changed` to the tool table, one line.
- `ecology://sources` resource: no change.

## Tests — `shared/tests/test_seasonal_changes.py` (offline, synthetic observations)

1. Species in `now` (3 obs) and not in `before` → one `arrived` with the right `first_seen`.
2. Species in `before` (3 obs) and not in `now` → one `departed` with the right `last_seen`.
3. Single sighting in `now` only → no change.
4. 2 → 12 observations → `more_often`; 2 → 4 → nothing; note contains "reported".
5. Arrived this year on Mar 10, first seen last year Mar 20 → `timing_vs_last_year_days == -10`, `last_year == "also_present"`.
6. `last_year=None` → every change has `last_year == "unknown"`.
7. `tally` merges the same species from two adapters into one entry with both sources.

`servers/orchestrator/tests/test_whats_changed.py` (offline, fake adapters):

8. Tool returns arrivals from faked iNaturalist data, includes the caveat, and caps `changes` at 25.
9. With fewer than 20 observations the caveat mentions the small sample.

## Do Not

- Do not add bird-song or photo identification. That is what Merlin and Seek
  do, on the phone, and neither is a data source here.
- Do not store exact home coordinates anywhere.
- Do not call a change a trend. Two windows are a comparison, not a pattern.
- Do not add a new adapter.

## Definition of Done

- "What changed near me in the last two weeks?" returns arrivals and
  departures with a plain sentence each and an honest caveat.
- Works with no API keys (iNaturalist only).

## Commit

```
Add ecology_whats_changed: arrivals and departures near home (spec 022)

- seasonal_changes: tally windows, compare now / before / last year
- ecology_whats_changed tool (iNaturalist, plus eBird when keyed)
- 9 tests
```
