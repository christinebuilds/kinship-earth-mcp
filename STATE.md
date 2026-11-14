# STATE — Kinship Earth MCP

_Last updated: 2026-11-14 by Claude (runner wake 6)_

## Where we are
Direction: `docs/DIRECTION.md` (read first). "Is this normal for here, and
what usually follows?", rivers first: Russian River plus Mission Creek.
This runner session can now push, read GitHub issues, and reach USGS,
Open-Meteo archive and NEON. `api.inaturalist.org` is still refused.
- Spec 015 (honest baselines) **done**. Flow is compared with one gauge's
  own day-of-year record. Live: Russian River at Guerneville (USGS-11467000),
  87 years, 179 cfs vs median 307 = 13th percentile, no anomaly. Climate
  normals cover 30 years (1996–2025) and report a trend.
- Specs 017 and 022 done earlier.

## Next up (no decision needed)
1. Each wake, first: triage open `bug` issues without `fix-ready` (none on
   2026-11-14).
2. Spec 016 (history replay + scorecard): `ready`. It needs a sourced list
   of past Russian River flow events (web search), then a no-look-ahead
   replay using `get_daily_record` (cached, so replay is cheap).
3. Spec 018 (Taylor's power law on NEON bird counts): `ready`, NEON reachable.
4. Issue #18 (`data-gap`: neighborhood flood outlook): find the open data,
   write a spec, file a Radar decision on whether to queue it.
5. Write specs 019–021 (satellite source cards, canon MCP tools, canon CI).
   Spec buffer is low: 2 ready.

## Waiting on Christine
Project Radar (https://claude.ai/artifact/4e5P3qeCmC1rctQUrn651Q):
- **Mission Creek gauge** (new): no USGS gauge in SF. Say "no gauge",
  use the nearest (a Marin creek 20 km away), or same watershed only?
  Until answered, the code uses the nearest gauge within 50 km with ≥10 years.
- PR #17 (bug tracking): she said merge, but its `test` check is still
  cancelled (checked 2026-11-14). Needs a green re-run by its author session.
- Add a GitHub token on Railway (her action).
- `api.inaturalist.org` on this environment's allowlist, so
  `ecology_whats_changed` can be tried live.

## Found along the way
- USGS daily API: `sortby` works on the first page only (later pages 400);
  sort locally. `time-series-metadata` lists each gauge's daily record with
  begin/end dates, the cheap way to pick a gauge.
- The old `discharge_cfs` key bug is gone: flow now comes from the gauge's
  daily record, not a pooled search that mixed in water temperature.
- 3 tests in the "offline" command fail here because OBIS and iNaturalist
  answer 403; same before and after spec 015.
- `EcologicalGraph` keeps one edge per node pair.

## Recently done
- 2026-11-14 — Spec 015 finished (live gauges, 30-yr normals); network Radar row applied
- 2026-11-14 — Spec 022 built: `ecology_whats_changed`
- 2026-11-13 — Merged main; INFORMS edge; offline half of spec 015

## How to resume
- Protocol: `CLAUDE.md`; queue: `specs/QUEUE.md`. Decisions: Project Radar.
- Offline tests: the `-k "not (...)"` command under Testing in `CLAUDE.md`.
  Live tests are marked `@pytest.mark.network`.
- Runner branch: `claude/runner-kinship-earth`. Never push to main.
