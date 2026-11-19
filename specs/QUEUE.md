# Spec Queue — Autonomous Nightly Sessions

> This file is the source of truth for what the nightly autonomous session should work on.
> Each session reads this file, picks the next `ready` spec, executes it, and marks it `done`.

## How It Works

1. Session starts, reads this file
2. Finds the first spec with status `ready`
3. Reads that spec file for detailed instructions
4. Executes the spec: build, test, commit, push
5. Updates this file: marks spec as `done`, adds completion notes
6. Opens a PR or pushes to the feature branch
7. Writes a burn-down summary as a commit message

## Queue Status

| # | Spec File | Status | Target Date | Completed | Notes |
|---|-----------|--------|-------------|-----------|-------|
| 1 | `001-conversation-storage.md` | done | 2026-04-16 | 2026-04-15 | Phase 3.2 |
| 2 | `002-auth-proxy.md` | done | 2026-04-17 | 2026-04-15 | Phase 3.1 |
| 3 | `003-auth-integration.md` | done | 2026-04-18 | 2026-04-15 | Phase 3.1 + 3.2 |
| 4 | `004-workflow-tools.md` | done | 2026-04-19 | 2026-04-15 | Phase 2.2 |
| 5 | `005-visualization-hints.md` | done | 2026-04-20 | 2026-04-15 | Phase 2.3 |
| 6 | `006-graph-scaffold.md` | done | 2026-04-21 | 2026-04-15 | Phase 4.1 |
| 7 | `007-graph-pipeline.md` | done | 2026-04-22 | 2026-04-15 | Phase 4.1 |
| 8 | `008-memory-tools.md` | done | 2026-04-16 | 2026-04-15 | Phase 4.2 |
| 9 | `009-supabase-proxy.md` | done | 2026-04-16 | 2026-04-15 | Phase 3.1 + 3.3 |
| 10 | `010-memory-ranking.md` | done | 2026-04-16 | 2026-04-15 | Phase 4.3 + integration |
| 11 | `011-ecosystem-state.md` | done | 2026-05-12 | 2026-05-11 | Phase 5.1 |
| 12 | `012-anomaly-detection.md` | done | 2026-05-13 | 2026-05-11 | Phase 5.2 |
| 13 | `013-event-synthesis.md` | done | 2026-05-14 | 2026-10-31 | Phase 5.3 |
| 14 | `014-new-data-sources.md` | done | 2026-05-15 | 2026-11-05 | Phase 5.4 |
| 15 | `015-honest-baselines.md` | blocked | 2026-11-13 | | Requires live USGS API call before building; blocked by 403 in cloud env |
| 16 | `016-history-replay-scorecard.md` | blocked | 2026-11-14 | | Depends on 015; also needs live USGS gauge metadata calls |
| 17 | `017-canon-schema.md` | done | 2026-11-15 | 2026-11-15 | Phase 6 — pattern files + graph extension; 8 tests; branch claude/spec-017-canon-schema |
| 18 | `018-first-node-taylor.md` | done | 2026-11-16 | 2026-11-19 | Phase 6 — NEON adapter, check runner, canon node, 3 tests; network test needs local run |
| 19 | `019-satellite-source-cards.md` | queued | | | Source cards + first greenness adapter |
| 20 | `020-canon-mcp-tools.md` | queued | | | canon_list/get/neighbors/check/propose |
| 21 | `021-canon-ci-and-contributing.md` | queued | | | CI validation + pattern/evidence/edge PR guide |

### Status Key
- `ready` — spec is written, detailed, and executable autonomously
- `queued` — placeholder, spec not yet written (session should skip and flag)
- `in_progress` — currently being worked on
- `done` — completed, committed, pushed
- `blocked` — cannot proceed, see notes

## Inventory

- **Specs ready:** 0
- **Specs blocked:** 2 (015, 016 — require USGS live API, blocked in cloud env)
- **Specs queued (need writing):** 3
- **Specs done:** 16
- **Buffer: 0 specs ready — WRITE NEW SPECS (019+) in next interactive session**

> When buffer drops to 2 or fewer, the next interactive session should prioritize writing more specs.
