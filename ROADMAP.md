# Kinship Earth — Roadmap

> Last updated: 2026-04-15
>
> See [STRATEGY.md](STRATEGY.md) for vision, architecture, and key technical decisions.

---

## Phase Overview

| Phase | Name | Focus | Status |
|---|---|---|---|
| 1 | Foundation | 9-source federation + unified schema | **Complete** |
| 2 | MCP Maturity | Prompt templates, workflow tools, export tools | **Next** |
| 3 | Auth & Persistence | Identity, conversation storage, feedback capture | Planned |
| 4 | Memory Graph | Knowledge graph, emergent patterns, cross-user intelligence | Planned |
| 5 | Ecosystem Intelligence | Anomaly detection, ecosystem state, continuous monitoring | Planned |

---

## Phase 1 — Foundation (Complete)

**Goal:** Make ecological data queryable by AI agents through a single MCP interface.

### Delivered
- [x] Unified `EcologicalObservation` schema (Darwin Core–aligned)
- [x] Adapter pattern with `EcologicalAdapter` base class
- [x] 9 data source adapters: NEON, OBIS, ERA5, eBird, iNaturalist, GBIF, Xeno-canto, USGS NWIS, SoilGrids
- [x] Orchestrator with 4 cross-source tools (`ecology_search`, `ecology_get_environmental_context`, `ecology_describe_sources`, `ecology_whats_around_me`)
- [x] Federated relevance ranking (geo × taxon × temporal × quality)
- [x] GeoJSON export
- [x] Full provenance + CARE principles tracking
- [x] Schema snapshot tool for upstream API monitoring
- [x] Jupyter notebook examples
- [x] QGIS integration guide
- [x] 36 tests (real API integration tests)
- [x] CLI launcher (`kinship-earth <server> [transport]`)
- [x] Phase 2/3 schema types: `EcologicalAnomaly`, `EcologicalEvent`, `EcosystemState`

---

## Phase 2 — MCP Maturity (Next)

**Goal:** Make the MCP server a first-class agentic tool — richer interactions, composable workflows, structured outputs that help agents generate better UI.

**Duration:** 3–4 weeks

### Milestone 2.1 — Prompt Templates & Resources
MCP prompts let agents invoke curated multi-step workflows with a single call.

- [x] `ecological_survey` prompt — comprehensive biodiversity + climate + soil report for a location
- [x] `species_report` prompt — deep dive on a single species: occurrences, climate correlation, audio, soil
- [x] `site_comparison` prompt — structured comparison of 2+ locations across all data sources
- [x] `data_export` prompt — guided export in CSV, GeoJSON, Markdown report, or BibTeX
- [x] MCP resource: `ecology://sources` — live registry of available data sources and their status

### Milestone 2.2 — Workflow Composition Tools
Higher-level tools that chain existing tools into research workflows.

- [x] `ecology_biodiversity_assessment` — chains search → climate → soil → synthesis for a location
- [x] `ecology_temporal_comparison` — "what changed here between date A and date B?"
- [x] `ecology_export` — tool that outputs structured CSV, GeoJSON, Markdown report, or BibTeX with DOIs
- [x] `ecology_cite` — given a set of observations, produce properly formatted citations

### Milestone 2.3 — Output Structure for Agent UI
Help agents render better visualizations without prescribing the UI.

- [x] Add `visualization_hint` field to all tool responses (e.g., `"map"`, `"timeseries"`, `"comparison_table"`, `"species_gallery"`)
- [x] Add `map_data` structured field with GeoJSON + suggested bounds + layer groupings
- [x] Add `chart_data` structured field with labeled axes, series, and suggested chart type
- [ ] Document output schema for agent/client developers

### Phase 2 Definition of Done
- All 4 prompts registered and functional in Claude Desktop
- Workflow tools return structured data that Claude can render as artifacts (maps, charts)
- Existing tests still pass; new tests for workflow tools
- README updated with new capabilities

---

## Phase 3 — Auth & Persistence

**Goal:** Add identity and conversation memory so the system can learn from usage.

**Duration:** 4–5 weeks

**Dependency:** Decision on hosting model (authenticated MCP proxy vs. platform service)

### Milestone 3.1 — Authenticated MCP Proxy
A thin service layer that wraps the MCP server with auth.

- [x] OAuth provider integration (Google and/or GitHub)
- [x] User identity model (Supabase or Auth0 + Postgres)
- [x] Authenticated MCP transport (SSE with auth headers)
- [x] Free tier usage metering (queries/day)
- [x] BYOK API key vault (user provides their own eBird key, etc.)

### Milestone 3.2 — Conversation Persistence
Store every agent conversation for future memory graph ingestion.

- [x] Conversation storage schema (who, when, what tools called, what returned, what feedback given)
- [x] Dual-write to local SQLite (dev) + Supabase/Postgres (prod)
- [x] Per-message feedback capture (thumbs up/down, correction, annotation)
- [x] Conversation replay / export for users
- [ ] Privacy controls: per-user opt-in for shared memory, data deletion

### Milestone 3.3 — Extract Web App Components
Migrate reusable pieces from `kinship-earth-web` into the MCP platform.

- [ ] Port auth flow from web app to proxy service
- [ ] Port conversation storage schema
- [ ] Port feedback widget logic (as MCP tool: `ecology_feedback`)
- [ ] Port export logic as MCP tools (already planned in 2.2)
- [ ] Archive `kinship-earth-web` repo (or mark as deprecated)

### Phase 3 Definition of Done
- Users can authenticate and use Kinship Earth MCP with persistent identity
- Conversations are stored and associated with user accounts
- Feedback is captured per interaction
- Web app functionality fully migrated or archived
- Privacy controls documented and functional

---

## Phase 4 — Memory Graph

**Goal:** Build the ecological knowledge graph that compounds from every conversation.

**Duration:** 6–8 weeks

**Dependency:** Phase 3 (need conversations + identity to feed the graph)

### Milestone 4.1 — Graph Infrastructure
- [x] Evaluate and select graph engine (Graphiti vs. custom Neo4j vs. Memgraph + pgvector)
- [x] Design entity ontology: Species, Location, Watershed, Ecosystem, Researcher, Query, Finding, Anomaly
- [x] Design relationship types: OBSERVED_AT, INHABITS, DRAINS_TO, CORRELATES_WITH, ASKED_ABOUT, SIMILAR_TO
- [x] Temporal edge model (validity windows, fact versioning — "temperature normal was X until 2024, now Y")
- [x] Implement graph write pipeline: conversation → entity extraction → graph upsert

### Milestone 4.2 — Memory-Aware MCP Tools
New tools that query the memory graph alongside live data.

- [x] `ecology_related_queries` — "3 other researchers queried this watershed last month, focusing on salmonid habitat"
- [x] `ecology_emerging_patterns` — "across all users, growing species/location mention counts and co-occurrence patterns"
- [x] `ecology_memory_store` — agent explicitly saves an insight to the graph ("researcher confirmed coho spawning at this tributary")
- [x] `ecology_memory_recall` — retrieve relevant memory nodes for a given location/species/time

### Milestone 4.3 — Memory-Informed Ranking
Existing search results get re-ranked based on graph context.

- [x] Add memory_relevance component to federated ranking score
- [x] Weight observations higher if the location/species has active research interest
- [ ] Surface "you might also want to know" connections from the graph
- [ ] A/B test: memory-informed vs. baseline ranking (measure user feedback signals)

### Phase 4 Definition of Done
- Knowledge graph populated from stored conversations
- Memory tools return meaningful results for locations/species with conversation history
- Ranking incorporates memory signals
- Cold-start strategy working (valuable even with sparse graph)
- Privacy model enforced (per-user vs. shared memory)

---

## Phase 5 — Ecosystem Intelligence

**Goal:** Move from reactive queries to proactive ecological monitoring and anomaly detection.

**Duration:** 8–12 weeks

**Dependency:** Phase 4 (graph provides baseline patterns for anomaly detection)

### Milestone 5.1 — Continuous Monitoring
- [ ] `EcosystemState` continuous computation for monitored locations
- [ ] Baseline calculation from ERA5 + NEON + USGS historical data
- [ ] Scheduled data pulls (cron or event-driven) for active monitoring sites

### Milestone 5.2 — Anomaly Detection
- [ ] Implement `EcologicalAnomaly` detection pipeline (schema already defined)
- [ ] Temperature, flow, phenological, acoustic, composition anomaly detectors
- [ ] Alert tool: `ecology_check_anomalies` — "any unusual signals at this location?"
- [ ] Anomaly → graph edge: anomalies become first-class graph entities

### Milestone 5.3 — Event Synthesis
- [x] Implement `EcologicalEvent` classification (schema already defined)
- [x] Multi-anomaly → event correlation (e.g., low flow + high temp + die-off = drought cascade)
- [x] Historical analog matching ("this looks like the 2021 Klamath drought cascade")
- [x] Proactive notifications: agent subscribes to locations and gets alerted

### Milestone 5.4 — Additional Data Sources
- [x] Movebank (animal tracking / migration data)
- [x] FLUXNET (carbon/water/energy flux towers)
- [ ] Wildlife Insights (camera trap imagery)
- [ ] Copernicus Land (satellite-derived land cover, NDVI)

### Phase 5 Definition of Done
- At least 5 monitored locations with continuous EcosystemState
- Anomaly detection firing on real data with <24hr latency
- Events synthesized from multi-signal anomalies
- Graph contains temporal ecological state history
- Agents can subscribe to location alerts

---

## Success Metrics

| Metric | Phase 2 Target | Phase 4 Target | Phase 5 Target |
|---|---|---|---|
| MCP tools available | 20+ | 28+ | 35+ |
| Data sources federated | 9 | 9+ | 13+ |
| Registered users | — | 50+ | 200+ |
| Stored conversations | — | 500+ | 5,000+ |
| Graph entities | — | 10,000+ | 100,000+ |
| Graph relationships | — | 50,000+ | 500,000+ |
| Avg. query response time | <5s | <5s | <5s |
| Memory-informed queries (% with graph hits) | — | 30%+ | 60%+ |

---

## Open Questions

1. **Hosting model:** Authenticated MCP proxy on Railway/Fly? Or platform service with managed MCP endpoints?
2. **Graph engine:** Graphiti (built for agent memory) vs. Neo4j (mature, rich query language) vs. Memgraph (performance)?
3. **Pricing model:** Free tier → paid? Per-query? Per-user? API key marketplace?
4. **Data governance:** How do we handle shared memory across users for indigenous land / CARE-restricted data?
5. **Multi-tenancy:** Separate graphs per organization, or one global graph with access controls?
