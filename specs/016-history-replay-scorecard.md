# Spec 016: History Replay and Scorecard

> Phase 6 — "Is this normal for here?" (see docs/DIRECTION.md)
> Priority: P0
> Estimated effort: 1–2 sessions
> Dependency: Spec 015

## Objective

Run the flow detector over past decades at a small set of gauges and publish how often it was right. The output is a scorecard: for documented low-flow and high-flow events, did the detector flag them, how early, and how many flags had no documented event behind them. This is the first published reliability record for any Kinship Earth detector.

## Why

Event rules and the six entries in `KNOWN_ANALOGS` are hand-written and have never been scored. Design principle 6 in docs/DIRECTION.md: publish hit rate and false-alarm rate for every detector.

## What to Build

### 1. Reference event list — `canon/events/reference_events.yaml`

A small, cited list of documented hydrological events at specific gauges. Each entry:

```yaml
- id: klamath-2021-low-flow
  kind: low_flow            # low_flow | high_flow
  site_id: USGS-11530500    # confirm the gauge id with a live metadata call
  start: 2021-06-01
  end: 2021-10-31
  title: 2021 Klamath Basin drought
  source: "<URL or DOI of an agency or peer-reviewed description>"
  note: "one sentence on what was documented"
```

- Start with **8–12 events across 4–6 gauges** in California and Oregon: droughts of 1976–77, 2012–16 and 2020–22, and floods such as early 1997, 2017 and 2023, at long-record gauges (Russian, Klamath, Eel, Sacramento, and one snowmelt-fed Sierra river).
- **Every event needs a real, checked source** (USGS, NOAA, state agency report, or a paper). Use web search to find and open each source. If a date range or gauge cannot be sourced, leave the event out. Do not write dates from memory.
- Confirm each `site_id` exists and has a daily record covering the event, using `get_daily_record` from spec 015.
- If Christine has named a first watershed in docs/DIRECTION.md "Open decisions", weight the list toward it. Otherwise use the Russian River as the default.

### 2. Replay engine — `shared/src/kinship_shared/replay.py`

```python
class ReplayDay(BaseModel):
    day: date; flow_cfs: float; percentile: float | None; flag: Literal["low", "high"] | None

def replay_flow(record: list[tuple[date, float]], start: date, end: date,
                window_days: int = 7, min_years: int = 10,
                low_pct: float = 10, high_pct: float = 90) -> list[ReplayDay]
```

- For each day, compute the percentile against the record **using only years before that day's year** (no future data). Days with fewer than `min_years` of prior record get `percentile=None`.
- Group consecutive flagged days into episodes; an episode needs ≥ 14 consecutive low days or ≥ 2 consecutive high days (make both parameters).

### 3. Scoring — same module

```python
class Scorecard(BaseModel):
    site_id: str; years_replayed: int
    events_total: int; events_hit: int
    lead_days_median: float | None       # first flag relative to documented start; negative = flagged before
    episodes_total: int; episodes_unmatched: int
    notes: list[str]
```

- A documented event is **hit** if a flagged episode of the matching kind overlaps its date range.
- An episode is **unmatched** if it overlaps no documented event. Label these "unmatched", not "false": the reference list is short, so many unmatched episodes will be real events nobody listed. Say this in the report.
- Report results per gauge and pooled.

### 4. CLI and report

`uv run python tools/replay_scorecard.py --events canon/events/reference_events.yaml --out canon/scorecards/`

Writes:

- `canon/scorecards/flow-detector.json` — the scorecards.
- `canon/scorecards/flow-detector.md` — a plain-language page: one table (gauge, years replayed, events hit / total, median lead, unmatched episodes per decade), one paragraph on what the numbers do and do not show, and the list of unmatched episodes longer than 60 days as candidates for the reference list.
- One figure per gauge, `canon/scorecards/figures/{site_id}.png`: flow percentile through time with documented events shaded and flagged episodes marked. Matplotlib only.

### 5. Threshold sensitivity

Run the scoring at low thresholds of 5, 10 and 20 and report all three in the table. Do **not** pick the threshold that looks best and present it alone.

## Tests — `shared/tests/test_replay.py` (offline)

1. Synthetic 50-year record with an injected 60-day drought in year 40: `replay_flow` flags it; scoring against a matching reference event gives 1/1 hit.
2. No look-ahead: changing values after day D does not change the percentile at D.
3. First `min_years` of the record produce `percentile=None` and no flags.
4. Episode grouping respects the minimum-length parameters.
5. An episode with no reference event is counted as unmatched, and the event count is unaffected.

## Do Not

- Do not tune thresholds to maximise the hit rate.
- Do not describe unmatched episodes as false alarms.
- Do not add events without an opened, cited source.
- Do not replay temperature, composition or phenology detectors yet (flow only).

## Definition of Done

- A committed reference list with sources, a scorecard JSON and Markdown page, and figures.
- README links the scorecard under a heading "How often is it right?"
- `docs/DIRECTION.md` "Open decisions" gains a line with the headline numbers and any surprises.
- Note in the commit message whether the Klamath 2021 event was flagged and with what lead.

## Commit

```
History replay and first scorecard for the flow detector (spec 016)

- canon/events/reference_events.yaml: N sourced events at M gauges
- replay.py: no-look-ahead percentile replay, episode grouping, scoring
- tools/replay_scorecard.py; scorecard JSON + Markdown + figures
- Klamath 2021: <flagged/not flagged>, lead <days>
- 5 tests
```
