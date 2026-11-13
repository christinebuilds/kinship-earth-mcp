# STATE — Kinship Earth MCP

_Last updated: 2026-11-13 by Claude (runner wake 3)_

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
`data.neonscience.org` (018). The `kinship-earth-mcp` environment allows them, but this
session is still on Default. Until a host allowlist exists, 015's
remaining half, 016 and 018 cannot run.

## Next up (no decision needed)
1. If the network allows: finish spec 015 — USGS `get_daily_record` (probe
   paging live first), single-gauge `compute_baselines_from_usgs`, 30-year
   ERA5 normals with trend, `state_builder` wiring, README sentences.
2. Otherwise build spec 022 (`ecology_whats_changed`); it is `ready` and
   needs no network.
3. Write specs 019–021 (satellite source cards, canon MCP tools, canon CI).
4. Specs 016 and 018 after 015 and network.
5. Each wake, first: triage open `bug` issues without `fix-ready` (none on
   2026-11-13). Christine allowed fix PRs, merged only after her yes on a
   `merge-fix-N` decision.

## Waiting on Christine
Project Radar (https://claude.ai/artifact/4e5P3qeCmC1rctQUrn651Q):
- Network: environment `kinship-earth-mcp` now exists with the three data
  hosts (approved 2026-11-13), but this runner session cannot move onto it.
  The runner needs to be restarted on that environment (new session or the
  wake routine pointed at it); until then the hosts stay blocked here.
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
- 2026-11-13 — Spec 022 written (neighborhood companion)
- 2026-11-13 — Merged main; INFORMS edge; offline half of spec 015
- 2026-11-12 — Pattern schema, loader, validator (now spec 017)

## How to resume
- Protocol: `CLAUDE.md`; queue: `specs/QUEUE.md`. Decisions: Project Radar.
- Offline tests: the `-k "not (...)"` command under Testing in `CLAUDE.md`.
- Runner branch: `claude/runner-kinship-earth`. Never push to main.
