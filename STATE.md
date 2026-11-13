# STATE — Kinship Earth MCP

_Last updated: 2026-11-13 by Claude (runner wake 2)_

## Where we are
Direction: `docs/DIRECTION.md` (read first). "Is this normal for here, and
what usually follows?", rivers first: Russian River plus Mission Creek in
San Francisco. This branch is `main` (through PR #16) plus the runner's work:
- Spec 017 (pattern schema, loader, graph extension) done, with `INFORMS`
  added as a tenth edge type.
- Spec 015 (honest baselines) half done: `FlowClimatology`,
  `flow_climatology`, `flow_percentile`, percentile-based `FlowDetector`,
  new `EcosystemState` fields, 9 offline tests.

## Blocked: network
The runner's cloud environment (Default) refuses every data API. Needed:
`api.waterdata.usgs.gov`, `archive-api.open-meteo.com` (015, 016) and
`data.neonscience.org` (018). An environment proposal was shown on
2026-11-13 and closed unanswered. Until a host allowlist exists, 015's
remaining half, 016 and 018 cannot run.

## Next up (no decision needed)
1. Once the network allows: finish spec 015 — USGS `get_daily_record`
   (probe paging live first), single-gauge `compute_baselines_from_usgs`,
   30-year ERA5 normals with trend, `state_builder` wiring, README sentences.
2. Write `specs/022-neighborhood-companion.md` (small; seasonal changes near
   home). Needs no network.
3. Write specs 019–021 (satellite source cards, canon MCP tools, canon CI).
4. Specs 016 and 018 after 015 and network.

## Waiting on Christine
Project Radar (https://claude.ai/artifact/4e5P3qeCmC1rctQUrn651Q):
- Allow the runner to reach the three data hosts above? (blocks all specs)
- Bug flow: the setup session relayed a procedure where the runner opens
  fix PRs and merges on a yes. The runner's standing rules say never open
  PRs or merge; it needs Christine's own word before it does either.

## Found along the way
- `state_builder` and the old USGS baseline read `value["discharge_cfs"]`,
  but the adapter writes `value["measurement"]`, so streamflow has been
  silently empty. Fix belongs in the rest of spec 015.
- Mission Creek is mostly culverted; it may have no USGS gauge. Check live.
- 7 tests in the "offline" command call live APIs and fail here.
- `EcologicalGraph` keeps one edge per node pair.
- Christine is the first pattern reviewer (her answer, 2026-11-13).

## Recently done
- 2026-11-13 — Merged main; INFORMS edge; offline half of spec 015
- 2026-11-12 — Pattern schema, loader, validator (now spec 017)

## How to resume
- Protocol: `CLAUDE.md`; queue: `specs/QUEUE.md`. Decisions: Project Radar.
- Offline tests: the `-k "not (...)"` command under Testing in `CLAUDE.md`.
- Runner branch: `claude/runner-kinship-earth`. Never push to main.
