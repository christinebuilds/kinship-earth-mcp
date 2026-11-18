# STATE — Kinship Earth MCP

_Last updated: 2026-11-18 by Claude (runner wake 9)_

## Where we are
Direction: `docs/DIRECTION.md` (read first). "Is this normal for here, and
what usually follows?", rivers first: Russian River plus Mission Creek.
This runner session can now push, read GitHub issues, and reach USGS,
Open-Meteo archive and NEON. `api.inaturalist.org` is still refused.
- Spec 015 (honest baselines) **done**. Flow is compared with one gauge's
  own day-of-year record. Live: Russian River at Guerneville (USGS-11467000),
  87 years, 179 cfs vs median 307 = 13th percentile, no anomaly. Climate
  normals cover 30 years (1996–2025) and report a trend.
- Same-watershed rule (her answer 2026-11-14): a gauge counts only if it
  shares the point's 10-digit USGS watershed code; otherwise
  `no_gauge_in_watershed`. Russian River places keep their gauges. Mission
  Creek still matches San Mateo Creek (27 km) because USGS draws one
  watershed around both; exact fix needs api.water.usgs.gov (Radar).
- CI split (her yes 2026-11-14): half done. 7 tests that call live
  services are marked `network`; the offline command now passes
  `-m "not network"` (190 pass). Editing `.github/workflows` was refused by
  this session's permission check, so the workflow is unchanged.
- Specs 017 and 022 done earlier.

## Next up (no decision needed)
1. Each wake, first: triage open `bug` issues without `fix-ready` (none on
   2026-11-18).
2. Spec 016, part 2: engine done (`shared/src/kinship_shared/replay.py`).
   Next: source 8–12 events with WebSearch/WebFetch into
   `canon/events/reference_events.yaml` (open every source; no dates from
   memory), then `tools/replay_scorecard.py`, the 5/10/20 threshold sweep,
   and the report. Figures need matplotlib added as a dev dependency.
   First look (Guerneville, 87 yrs): long low episodes in 1977, 2009,
   2014–15, 2020–22; 346 high episodes of ≥2 days (worth a note).
3. Spec 018 (Taylor's power law on NEON bird counts): `ready`, NEON reachable.
4. Issue #18 (`data-gap`: neighborhood flood outlook): find the open data,
   write a spec, file a Radar decision on whether to queue it.
5. Write specs 019–021 (satellite source cards, canon MCP tools, canon CI).
   Spec buffer is low: 2 ready.

## Waiting on Christine
Project Radar (https://claude.ai/artifact/4e5P3qeCmC1rctQUrn651Q):
- **CI workflow edit**: needs a session allowed to edit `.github/workflows`.
  Plan: `test.yml` runs the offline command with `-m "not network"`;
  a new `nightly.yml` runs the full suite on a schedule (60 min limit).
- **api.water.usgs.gov**: she said yes (2026-11-17), but the approval
  prompt closed unanswered and the host is still refused. Radar row says
  how to add it in Settings. Connected-by-water check waits on it.
- PR #17 (bug tracking): her re-run was cancelled at 15 min again
  (2026-11-14). She can merge it as owner, or it waits for the CI split.
- Add a GitHub token on Railway (her action).
- `api.inaturalist.org` on this environment's allowlist, so
  `ecology_whats_changed` can be tried live.

## Found along the way
- USGS daily API: `sortby` works on the first page only (later pages 400);
  sort locally. `time-series-metadata` lists each gauge's daily record with
  begin/end dates, the cheap way to pick a gauge.
- The old `discharge_cfs` key bug is gone: flow now comes from the gauge's
  daily record, not a pooled search that mixed in water temperature.
- Watershed codes come from the nearest USGS site within 10 km (the
  national boundary service hydro.nationalmap.gov is blocked here).
- `test_whats_around_me.py` still calls live sightings services in the
  offline set (passes slowly, ~100 s); mark it `network` with the CI split.
- `EcologicalGraph` keeps one edge per node pair.

## Recently done
- 2026-11-18 — Spec 016 part 1: replay engine and scoring, 5 tests
- 2026-11-16 — Same-watershed gauge rule; live tests marked `network`
- 2026-11-14 — Spec 015 finished (live gauges, 30-yr normals); network Radar row applied
- 2026-11-14 — Spec 022 built: `ecology_whats_changed`
- 2026-11-13 — Merged main; INFORMS edge; offline half of spec 015

## How to resume
- Protocol: `CLAUDE.md`; queue: `specs/QUEUE.md`. Decisions: Project Radar.
- Offline tests: the `-k "not (...)"` command under Testing in `CLAUDE.md`.
  It now includes `-m "not network"`; live tests carry that marker.
- Runner branch: `claude/runner-kinship-earth`. Never push to main.
