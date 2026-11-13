# Kinship Earth MCP — Development Notes

## Direction — read first (updated 2026-11-12)

**Read `docs/DIRECTION.md` before anything else.** It records the November 2026 decisions and supersedes parts of STRATEGY.md.

In short:
- The product question is **"Is this normal for here, and what usually follows?"**, answered from open data, with a published record of how often the answer has been right.
- **Non-commercial.** No pricing or tier work.
- **Patterns, not laws.** Track where each pattern holds and how it is drifting.
- **First use case: rivers and watersheds.** Fix the baselines (spec 015), then replay history and publish a scorecard (spec 016). The canon schema and first pattern check follow (017, 018).
- **Data must be easy to get at scale:** satellite, open sightings, public science databases. Satellite is the first new source to scout.
- Users in mind: cities planning for El Niño winters, rural fire districts in Northern California, grassfed ranchers, Point Blue Conservation Science, and scientists without sensor or compute budgets.

When talking with Christine: plain language, bottom line first, one question at a time. She built PastureMap for 8 years and knows ranching and rancher data well.

## Session Ritual (TPM / Chief of Staff)
- **Start of session:** Review progress against ROADMAP.md and specs/QUEUE.md. Report: what phase/milestone we're in, what's on track, what's at risk, what's blocked. Present the **Decisions Needed** list below. Keep us accountable against the timeline.
- **During session:** Track deliverables against milestone checklists. Flag scope creep or distractions.
- **End of session:** Summarize what was completed, update ROADMAP.md checkboxes and specs/QUEUE.md status, note blockers, state next priorities. Report spec buffer ("X specs ready, Y nights of runway").

## Decisions Needed From Christine

These decisions are blocking or will soon block autonomous progress. Review and decide at your convenience — I'll incorporate your answers into the next specs.

### P0 — Direction (added 2026-11-12)

A. **Which watershed first?** Default is the Russian River (the code already names it). Not yet confirmed.
B. **Two domain reviewers.** A pattern cannot be marked `supported` without a human reviewer. Point Blue is the first place to ask.
C. **Rewrite STRATEGY.md in full?** It carries a "superseded" banner for now.
D. **Merge `claude/canon-foundation` into `main`.** It holds docs/DIRECTION.md, docs/CANON.md and specs 015–018.

### P0 — Blocking Next Specs (from April; items 1–3 are decided, see "Decided" below)

1. **Auth provider: Supabase vs Auth0 vs Clerk?**
   - Web app used Supabase. Stick with it for consistency, or switch?
   - Affects: spec for OAuth flow, user management, conversation storage prod backend
   - My recommendation: Supabase (you already have it, it's cheaper, it does auth + DB + storage)

2. **Pricing model: free tier limit?**
   - Current default: 50 queries/day free tier. Is that right?
   - Do we want a "pro" tier? What does it unlock?
   - Affects: rate limiting logic, BYOK key management

3. **Shared memory: opt-in or opt-out?**
   - When a user queries "dolphins near Woods Hole," should that contribute to a shared knowledge graph that other users can benefit from?
   - Privacy implications for indigenous land data (CARE principles)
   - Options: (a) all memory is per-user only, (b) opt-in shared, (c) shared by default with opt-out
   - My recommendation: opt-in shared, per-user by default

4. **kinship-earth-web: keep as demo client** (DECIDED)
   - Gut the duplicated orchestration (~300 lines), point at service/ API
   - Keep Leaflet map + chat UI as a lightweight public demo for non-agent users
   - Not archiving — becomes the "showroom" while MCP is the "engine"

### P1 — Needed Within 1-2 Sessions

5. **Domain/deployment: where does the authenticated MCP proxy live?**
   - Railway (current web app host)? Fly.io? Vercel edge functions?
   - Need this for prod Supabase integration and SSE transport

6. **Graph engine scale-up threshold:**
   - Currently using networkx + SQLite (good to ~100K entities)
   - At what scale do we migrate to Neo4j or Graphiti?
   - Or: do we just scale SQLite until it breaks?

7. **What ecological questions should the memory graph prioritize?**
   - "What researchers are working on this watershed?"
   - "Has this species range shifted over time?"
   - "What locations have unusual recent activity?"
   - Helps me prioritize which memory tools to build first

### P2 — Future Planning

8. **Additional data sources priority order:**
   - Movebank (animal tracking), FLUXNET (carbon flux), Wildlife Insights (camera traps), Copernicus Land (satellite)
   - Which matters most for your users?

9. **Target user personas:**
   - Academic researchers? Land managers? Citizen scientists? All three?
   - Affects: tool naming, default parameters, prompt templates

## Autonomous Session Protocol

When starting an autonomous session (no human present):

1. Read `specs/QUEUE.md` — find the first spec with status `ready`
2. Read the spec file — it contains everything: what to build, what to test, what to commit
3. **Execute the spec exactly as written** — create files, modify files, write tests
4. **Run all tests** — both new tests from the spec AND existing tests (validation/offline only)
5. **Commit with the template** from the spec, push to the working branch
6. **Update specs/QUEUE.md** — mark the spec as `done`, add completion date
7. **Update ROADMAP.md** — check off completed items
8. **Check spec buffer** — if 2 or fewer specs remain `ready`, note this in commit message so the next interactive session knows to write more specs

### If a spec is blocked:
- Mark it as `blocked` in QUEUE.md with a note explaining why
- Move to the next `ready` spec
- Do NOT skip queued (unwritten) specs — stop and flag

### Branch strategy:
- Work on branch `claude/canon-foundation` until Christine merges it into `main`; after that, branch from `main`. (The older `claude/mcp-chat-integration-8Tpu4` branch is behind `main` and does not have specs 015+.)
- Push after each spec completion
- Do not create PRs autonomously — the human will review and merge

## Architecture
- Monorepo: `shared/` (schema, ranking, adapters), `servers/` (one per data source + orchestrator), `launcher/`
- Framework: FastMCP (Python)
- Schema: `EcologicalObservation` in `shared/src/kinship_shared/schema.py` — Darwin Core-aligned
- All adapters implement `EcologicalAdapter` interface from `shared/src/kinship_shared/adapter.py`
- Orchestrator tools live in `shared/src/kinship_shared/ecology_tools.py`, server in `servers/orchestrator/`
- Conversation storage: `shared/src/kinship_shared/storage*.py`
- Knowledge graph: `shared/src/kinship_shared/graph*.py`
- Auth: `servers/orchestrator/src/kinship_orchestrator/auth*.py`

## Key Docs
- `docs/DIRECTION.md` — **Current direction and decisions (Nov 2026). Read first.**
- `docs/CANON.md` — The pattern library: file format, seed list, collaboration model
- `docs/game-concept.md` — Children's game concept (for fun, not scheduled)
- `STRATEGY.md` — Architecture and federation (moat and pricing sections superseded)
- `ROADMAP.md` — 5-phase plan with milestones and checklists
- `specs/QUEUE.md` — Ordered spec queue for autonomous sessions
- `specs/001-*.md` through `specs/018-*.md` — Detailed build specs

## Testing
```bash
# Full test suite (requires network for API tests)
uv run --package kinship-orchestrator pytest servers/ shared/tests/ -v

# Offline-only tests (always runnable)
uv run --package kinship-orchestrator pytest servers/orchestrator/tests/ shared/tests/ -v -k "not (climate or dolphin or cetac or wind_river or woods_hole or parallel or cross_persona or marine or reachable or coordinates or geographic or bird_survey or bird_data or catalog or nonexistent or empty_area or bogus)"
```

## Current Status (Updated 2026-11-12)

### November 2026
- Phase 5 complete through 5.4 (Movebank and FLUXNET adapters landed 2026-11-05; 11 sources federated)
- Direction reset recorded in `docs/DIRECTION.md`
- Phase 6 specs 015–018 written and `ready`; none built yet
- Known weaknesses to fix first: streamflow baseline is the last 30 days pooled across gauges; climate baseline is 5 years; flow anomalies use percent-of-mean; event rules and analogs are hand-written and unscored

### Earlier status (as of 2026-04-15)

### Completed
- Phase 1: 9-source federation (DONE)
- Phase 2.1: Prompt templates + ecology://sources resource (DONE)
- Phase 2.2: Workflow tools — biodiversity assessment, temporal comparison, export, cite (DONE)
- Phase 2.3: Visualization hints — map, timeseries, gallery, auto-select (DONE)
- Phase 3.1: Auth — user model, rate limiting, BYOK keys (DONE, needs OAuth flow)
- Phase 3.2: Conversation storage — SQLite, feedback, history (DONE, needs Supabase)
- Phase 4.1: Graph scaffold — entity ontology, graph store, extraction pipeline (DONE)

### In Progress
- Phase 4.3: "You might also want to know" suggestions (partial)
- Phase 5: Ecosystem intelligence (next phase)

### Decided
- Web app stays as demo client, points at service/ API
- Supabase for auth
- Opt-in shared memory, per-user by default
- Non-commercial (2026-11-12): no pricing tiers; BYOK only so users can cover their own API costs

### Server Stats
- **17 tools**, **4 prompts**, **1 resource**
- **~103 tests** across all modules (all passing)
- Tracking issues: #5 (done), #6 (done), #7 (done), #8 (done), #9 (done), #10 (done)
