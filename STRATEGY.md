# Kinship Earth — Technical Strategy

> Last updated: 2026-04-15
>
> **Partly superseded 2026-11-12 — see docs/DIRECTION.md.** Kinship Earth is non-commercial, so the pricing and free/pro tier material no longer applies. The "Competitive Moat" section (proprietary conversation corpus, cross-user emergence as moat) is retired: the federation is a public verifier, and per-user memory stays opt-in. The architecture and federation sections still describe the code.

## Vision

Kinship Earth is an **ecological memory system** — a living knowledge graph that gets smarter every time a researcher, land manager, student, or AI agent asks a question about the natural world.

Today we federate 9 ecological APIs into a unified intelligence layer. Tomorrow, every conversation compounds into a shared graph of ecological understanding that no single dataset, tool, or institution could produce alone.

The product is not a web app. The product is the intelligence layer — consumed by Claude, Cursor, Jupyter, QGIS, and whatever agentic interfaces emerge next.

---

## Core Thesis

1. **AI agents will generate UIs on the fly.** Maps, charts, dashboards, and reports will be produced dynamically by agents in response to specific questions. Building a fixed web app for visualization is building on sand.

2. **The durable moat is the intelligence layer + memory graph.** Public APIs are commodities. Unified schema, cross-source ranking, provenance tracking, and emergent ecological memory are not.

3. **Conversations are training data.** Every query a researcher makes — "what species are near this watershed?", "how has temperature changed here since 2020?" — creates nodes and edges in an ecological knowledge graph. Cross-user patterns emerge that no individual could see.

4. **Identity enables memory.** Without auth, every session starts from zero. With identity, we build persistent context per user and per location, and compound intelligence across the community.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                         │
│  Claude Chat · Claude Code · Cursor · Jupyter · QGIS   │
│  (Agents generate UI — maps, charts, reports — on fly)  │
└────────────────────────┬────────────────────────────────┘
                         │ MCP Protocol
┌────────────────────────▼────────────────────────────────┐
│              MEMORY & GRAPH LAYER (new)                  │
│                                                          │
│  Conversation → Entity/Relationship Extraction           │
│  Temporal Fact Tracking (validity windows)                │
│  Cross-User Pattern Emergence                            │
│  Location Memory ("what's been asked about this place?") │
│  Ecological Relationship Graph (species ↔ habitat ↔      │
│    climate ↔ watershed ↔ researcher)                     │
│                                                          │
│  Tools:                                                  │
│    ecology_related_queries — "others asked about..."     │
│    ecology_location_history — "this place over time"     │
│    ecology_emerging_patterns — "cross-user signals"      │
│    ecology_memory_store — persist conversation insights  │
│    ecology_memory_recall — retrieve relevant memory      │
│                                                          │
│  Stack: Graphiti / Neo4j / pgvector                      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│            MCP INTELLIGENCE LAYER (exists)                │
│                                                          │
│  9-Source Federation                                     │
│  ├── NEON (81 US sites, 180+ data products)              │
│  ├── OBIS (168M+ marine records)                         │
│  ├── ERA5 (global climate, 1940–present)                 │
│  ├── eBird (1.5B+ bird observations)                     │
│  ├── iNaturalist (200M+ all-taxa observations)           │
│  ├── GBIF (2.8B+ occurrence records)                     │
│  ├── Xeno-canto (1M+ bird audio recordings)              │
│  ├── USGS NWIS (13,500+ US stream gauges)                │
│  └── SoilGrids (global soil, 250m resolution)            │
│                                                          │
│  Unified Schema (Darwin Core–aligned)                    │
│  Federated Ranking (geo × taxon × temporal × quality)    │
│  Provenance & CARE Principles                            │
│  GeoJSON Export                                          │
│  MCP Prompt Templates (new)                              │
│  Workflow Compositions (new)                             │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│             AUTH & PERSISTENCE LAYER (new)                │
│                                                          │
│  OAuth (Google/GitHub) — user identity                   │
│  Conversation Storage — raw material for memory graph    │
│  Feedback Capture — human signal on what matters         │
│  Usage Metering — free tier + BYOK API keys              │
│  API Key Vault — secure storage for eBird, etc.          │
│                                                          │
│  Stack: Supabase (auth + storage) or Auth0 + Postgres    │
└──────────────────────────────────────────────────────────┘
```

---

## What We Keep From kinship-earth-web

The web app prototype (FastAPI + vanilla JS, ~7,800 lines) validated key ideas. We **extract** the following into the MCP platform and **retire** the web app as a standalone product:

### Extract and integrate
| Component | Why it matters |
|---|---|
| Google OAuth + Supabase auth | Identity is prerequisite for memory |
| Conversation persistence (SQLite + Supabase) | Raw material for the memory graph |
| Per-message feedback widget | Human signal for graph edge weighting |
| Usage metering + BYOK key management | Business model infrastructure |
| Data export (CSV, GeoJSON, Markdown, BibTeX) | Add as MCP tools, not web-only features |

### Retire (already in MCP or agents will replace)
| Component | Why |
|---|---|
| `_run_search()` / `_run_env_context()` orchestration | Duplicates MCP orchestrator (~300 lines) |
| Tool definitions + multi-source fan-out | Exact copy of MCP logic |
| Result normalization + scoring | Uses `kinship_shared` already |
| Leaflet map rendering | Agents generate maps dynamically via artifacts |
| SSE streaming UI with source badges | Agent UIs will handle streaming natively |
| Marketing/vision static pages | Move to a simple landing page or docs site |

---

## Competitive Moat

| Layer | Defensibility | Status |
|---|---|---|
| **Unified Schema** | Medium — hard to replicate well, but technically possible | Built |
| **9-Source Federation** | Medium — adapter pattern makes this achievable but labor-intensive | Built |
| **Federated Ranking** | Medium — domain expertise encoded in weights and scoring | Built |
| **Memory Graph** | **High** — network effect, every user makes it smarter | Not built |
| **Conversation Corpus** | **High** — proprietary dataset of ecological questions + answers | Partially built (web app) |
| **Cross-User Emergence** | **Very High** — patterns only visible across many researchers' queries | Not built |

The memory graph is the moat. Everything else is the foundation that makes it possible.

---

## Key Technical Decisions

### Why MCP, not a REST API?
- MCP is the native protocol for AI agents. Claude, Cursor, Windsurf, and others speak it natively.
- Tools are self-describing — agents discover capabilities without documentation.
- Composable — agents chain tools, inject context, and adapt to the question.
- REST is for apps. MCP is for agents. We're building for agents.

### Why not build our own UI?
- Agent-generated UI will surpass static dashboards within months, not years.
- Claude artifacts already render React, Leaflet maps, D3 charts, and interactive components inline.
- Every hour spent on frontend is an hour not spent on the intelligence layer.
- If we need a public-facing experience, a thin client calling MCP is sufficient.

### Why a memory graph, not just vector search?
- Vector search finds similar content. Graphs find **relationships**.
- "This watershed was studied by 3 researchers last month" is a graph query, not a similarity search.
- Temporal validity (facts that change over time) requires graph edges with time windows.
- Emergent patterns (species range shifts, phenological mismatches) are subgraph patterns.
- We'll use both: pgvector for semantic retrieval, graph for relationship reasoning.

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| MCP protocol evolves rapidly | Medium | Adapter pattern isolates protocol changes; FastMCP handles most |
| Upstream APIs change schemas | Medium | Schema snapshot tool already monitors; adapters absorb changes |
| Memory graph cold-start | High | Seed with existing web app conversations; offer immediate value without memory via federation layer |
| Privacy/CARE compliance for shared memory | High | Per-user memory by default; opt-in shared memory; CARE principles already in schema |
| Agent UI generation not mature enough | Low | Trajectory is clear; MCP tools work even without rich UI |
