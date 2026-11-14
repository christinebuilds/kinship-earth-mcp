# STATE — Kinship Earth MCP

_Last updated: 2026-11-14 by Claude (runner wake 4)_

## Where we are
Direction: `docs/DIRECTION.md` (read first). "Is this normal for here, and
what usually follows?", rivers first: Russian River plus Mission Creek in
San Francisco. This branch is `main` (through PR #16) plus the runner's work:
- Spec 017 (pattern schema, loader, graph extension) done, with `INFORMS`
  added as a tenth edge type.
- Spec 015 (honest baselines) half done: `FlowClimatology`,
  `flow_climatology`, `flow_percentile`, percentile-based `FlowDetector`,
  new `EcosystemState` fields, 9 offline tests.
- Spec 022 done: `ecology_whats_changed` (arrivals and departures near
  home), tested with faked adapters only.

## Blocked: network
The runner's cloud environment (Default) refuses every data API. Needed:
`api.waterdata.usgs.gov`, `archive-api.open-meteo.com` (015, 016) and
`data.neonscience.org` (018). The `kinship-earth-mcp` environment allows them, but this
session is still on Default. Until a host allowlist exists, 015's
remaining half, 016 and 018 cannot run.

## Next up (no decision needed)
1. Each wake, first: triage open `bug` issues without `fix-ready` (none on
   2026-11-14).
2. If the network allows: finish spec 015 — USGS `get_daily_record` (probe
   paging live first), single-gauge `compute_baselines_from_usgs`, 30-year
   ERA5 normals with trend, `state_builder` wiring, README sentences. Then
   try `ecology_whats_changed` once against live iNaturalist.
3. Otherwise: issue #18 (`data-gap`: neighborhood flood outlook) — find the
   open data, write a spec, file a Radar decision on whether to queue it.
4. Write specs 019–021 (satellite source cards, canon MCP tools, canon CI).
5. Specs 016 and 018 after 015 and network.

## Waiting on Christine
Project Radar (https://claude.ai/artifact/4e5P3qeCmC1rctQUrn651Q):
- Network: marked done in the Radar, but on 2026-11-14 the runner session
  is still on Default and every data host is refused. The wake still lands
  in this old session, which cannot be moved. It needs a new session on
  `kinship-earth-mcp` with the wake routine pointed at that session.
  `api.inaturalist.org` should be added to that environment too.
- PR #17 (bug tracking): she said merge, but GitHub blocks it — its `test`
  check was cancelled at 15 minutes. Needs a green re-run by the PR's author
  session before the runner merges.
- Add a GitHub token on Railway (her action).

## Found along the way
- `state_builder` and the old USGS baseline read `value["discharge_cfs"]`,
  but the adapter writes `value["measurement"]`, so streamflow has been
  silently empty. Fix belongs in the rest of spec 015.
- Mission Creek is mostly culverted; it may have no USGS gauge. Check live.
- 7 tests in the "offline" command call live APIs and fail here.
- `EcologicalGraph` keeps one edge per node pair.
- Christine is the first pattern reviewer (her answer, 2026-11-13).

## Recently done
- 2026-11-14 — Spec 022 built: `ecology_whats_changed`
- 2026-11-13 — Spec 022 written (neighborhood companion)
- 2026-11-13 — Merged main; INFORMS edge; offline half of spec 015

## How to resume
- Protocol: `CLAUDE.md`; queue: `specs/QUEUE.md`. Decisions: Project Radar.
- Offline tests: the `-k "not (...)"` command under Testing in `CLAUDE.md`.
- Runner branch: `claude/runner-kinship-earth`. Never push to main.
