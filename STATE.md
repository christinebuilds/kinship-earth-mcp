# STATE — Kinship Earth MCP

_Last updated: 2026-11-12 by Claude (runner wake 1)_

## Where we are
Specs 001–015 are done. Phase 6 (the Kinship Canon, `docs/CANON.md`) has its
foundation: `canon/` directory, `CanonNode` schema + loader, `pattern` entity
and nine evidential edge types in the graph, and `tools/canon_validate.py`.
No real nodes yet. Spec 016 is `ready`; 017 and 018 are unwritten.
This branch is `main` plus the canon-foundation commits and spec 015.

## Next up (no decision needed)
1. Execute `specs/016-first-node-taylor.md` (Taylor's power law on NEON
   breeding-bird counts). Everything except the `density-dependence` edge
   can proceed; that edge waits on the INFORMS decision below. Needs live
   NEON access for the snapshot pull — if the network blocks it, mark blocked.
2. Write specs 017 (canon MCP tools) and 018 (canon CI + contributing guide);
   both are `queued` placeholders in `specs/QUEUE.md`.
3. Refresh "Current Status" in `CLAUDE.md` (dated 2026-04-15, predates
   Phase 5 and the Canon).

## Waiting on Christine
See Notion → Home Stack → Decisions, Project = Kinship Earth:
- Edge type `INFORMS` is in the Taylor node draft but not in the schema:
  add it as a tenth type, remap, or drop the edge? (blocks part of 016)
- Retire the proprietary conversation-corpus "moat" framing?
- Who are the first two domain reviewers for the Canon?
- Merge `claude/canon-foundation` into main, or keep it as a branch?

## Known issues (not decisions)
- 7 tests in the "offline" command fail with or without recent changes
  (`test_cross_source` x4, `test_scientist_scenarios` x2,
  `test_whats_around_me::test_sf_has_observations`): they hit live APIs and
  are not excluded by the `-k` filter in `CLAUDE.md`.
- `EcologicalGraph` keeps one edge per node pair, so two canon edges of
  different types to the same target collapse to one in the graph.
- Spec 015 says `GraphStore`; the class is `EcologicalGraph`. Check results
  count as passing on `pass: true` (spec 016's key) or `passed: true`.
- CANON.md's node draft says evidence `source: neon`; the adapter id is
  `neonscience`. Spec 016 should use the adapter id so graph ids line up.

## Recently done
- 2026-11-12 — Canon schema, loader, graph ontology, validator CLI (spec 015)
- 2026-11-12 — STATE.md added; multicore protocol set up
- 2026-11-06 — Canon foundation doc, specs 015/016, Field Notebook game concept
- 2026-11-05 — Movebank and FLUXNET adapters (spec 014)

## How to resume
- Protocol: "Autonomous Session Protocol" in `CLAUDE.md`, queue in `specs/QUEUE.md`.
- Offline tests: the `-k "not (...)"` command under Testing in `CLAUDE.md`.
- Validate canon: `uv run --package kinship-shared python tools/canon_validate.py canon/patterns/`
- Runner branch: `claude/runner-kinship-earth`. Never push to main; no PRs.
