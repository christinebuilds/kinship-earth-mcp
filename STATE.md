# STATE — Kinship Earth MCP

_Last updated: 2026-11-27 by Claude (runner wake 16)_

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
   2026-11-27).
2. Spec 016 is blocked on drought sources (Radar). Built: replay engine,
   scoring (data gaps set aside, not counted as misses), CLI
   `tools/replay_scorecard.py`, figures. `canon/events/reference_events.yaml`
   has 11 flood peaks from USGS peak records. Preview (not published):
   10/10 floods flagged 2–30 days early; ~5 unmatched episodes per river
   per year, mostly 2-day winter highs. Klamath 1997 has no daily data.
   When hosts open: source droughts (Russian 1977/2014–15/2021, Klamath
   2021), run the CLI, commit `canon/scorecards/`, README "How often is it
   right?", DIRECTION.md line.
3. Spec 018 built, blocked on NEON token (Radar `kinship-neon-token`).
   When it lands: `uv run --package kinship-orchestrator python
   canon/checks/taylors_power_law.py --year 2023` (caches CSVs under
   canon/snapshots/neon/, writes result JSON, figure, updates node), commit
   snapshot + figure, README paragraph, then ask Christine to review the
   node (status stays `conjecture` until she does).
4. Issue #18: spec 023 written (`proposed`), Radar asks whether to queue it.
5. Spec 021 parts 1–2 done (canon_ci.py, contributor guide). Part 3, the
   workflow files, needs a session allowed to edit `.github/workflows`; the
   exact YAML is in the spec and in this file's Waiting list.
6. Spec 019 (satellite cards + MODIS greenness) written, blocked on hosts.
7. Build spec 024 next (streamflow recession law node): `ready`, needs
   only USGS, which works. First river pattern; tracks drift by decade.

## Waiting on Christine
Project Radar (https://claude.ai/artifact/4e5P3qeCmC1rctQUrn651Q):
- **CI workflow edit**: needs a session allowed to edit `.github/workflows`.
  Plan: `test.yml` runs the offline command with `-m "not network"`;
  a new `nightly.yml` runs the full suite on a schedule (60 min limit).
- **NEON token** (new): network secret for data.neonscience.org.
- **Drought sources**: she said allow (2026-11-19); hosts still refused.
  Radar note lists all hosts for one Settings visit.
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
- 2026-11-27 — Spec 024 written (recession law; GBIF/OBIS also blocked)
- 2026-11-26 — Spec 021 parts 1–2: canon CI checks, contributor guide
- 2026-11-25 — Spec 020 built: five canon tools, recall by pattern name
- 2026-11-24 — Specs 019 (blocked on hosts) and 020 (ready) written
- 2026-11-23 — Spec 023 written (flood outlook, from issue #18)
- 2026-11-20 — Spec 018 built offline (NEON download, Taylor check, node file)
- 2026-11-19 — Spec 016: sourced flood events, scorecard CLI; held for droughts
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
