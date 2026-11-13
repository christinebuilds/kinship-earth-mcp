# STATE — Kinship Earth MCP

_Last updated: 2026-11-12 by Claude (multicore setup session)_

## Where we are
Specs 001–014 are done; the last two shipped Oct 31 (event synthesis) and
Nov 5 (Movebank + FLUXNET adapters). Phase 6, the Kinship Canon, is designed
but not built: `docs/CANON.md` plus specs 015 and 016 are written and `ready`.
This branch is `main` plus the two `claude/canon-foundation` commits.

## Next up (no decision needed)
1. Execute `specs/015-canon-schema.md` (canon schema).
2. Execute `specs/016-first-node-taylor.md` (first canon node: Taylor's power
   law against NEON breeding-bird counts). Depends on 015.
3. Write specs 017 (canon MCP tools) and 018 (canon CI + contributing guide);
   both are `queued` placeholders in `specs/QUEUE.md`.
4. Refresh "Current Status" in `CLAUDE.md` (dated 2026-04-15, predates
   Phase 5 and the Canon).

## Waiting on Christine
See Notion → Home Stack → Decisions, Project = Kinship Earth:
- Retire the proprietary conversation-corpus "moat" framing?
- Who are the first two domain reviewers for the Canon?
- Merge `claude/canon-foundation` into main, or keep it as a branch?

## Recently done
- 2026-11-12 — STATE.md added; multicore protocol set up
- 2026-11-06 — Canon foundation doc, specs 015/016, Field Notebook game concept
- 2026-11-05 — Movebank and FLUXNET adapters (spec 014)
- 2026-10-31 — Event synthesis pipeline (spec 013)

## How to resume
- Protocol: "Autonomous Session Protocol" in `CLAUDE.md`, queue in `specs/QUEUE.md`.
- Offline tests: the `-k "not (...)"` command under Testing in `CLAUDE.md`.
- Never push to main; no PRs unless Christine asks.
