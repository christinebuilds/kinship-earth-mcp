# Kinship Earth

**Ecological intelligence for AI agents.**

Kinship Earth is a unified MCP server that makes ecological data queryable by AI. Ask Claude about species, climate, and ecosystems — and get answers that combine data from multiple scientific sources in a single response.

```
You: What were the climate conditions at Wind River Experimental Forest
     during the 2023 breeding season?

Claude: [calls ecology_get_environmental_context]

     Wind River (NEON site WREF, 45.82°N 121.95°W, 371m elevation)
     June 12-18, 2023:
     - Temperature: 12-28°C (mean 19.4°C)
     - Precipitation: 2.1mm total (dry week)
     - Soil temperature (0-7cm): 14.2°C mean
     - Soil moisture: 0.31 m³/m³
     - Nearest NEON site: WREF (0km) — 156 data products available
```

No single API gives you this. Kinship Earth combines **NEON** (81 US ecological sites), **OBIS** (168M+ marine species records), and **ERA5** (global climate from 1940) into one coherent interface.

---

## Quick Start (Claude Desktop)

### Prerequisites

- [Claude Desktop](https://claude.ai/download) installed
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) installed

### 1. Clone and install

```bash
git clone https://github.com/christinebuilds/kinship-earth-mcp.git
cd kinship-earth-mcp
uv sync
```

### 2. Add to Claude Desktop

Open your Claude Desktop config file:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add the Kinship Earth orchestrator server:

```json
{
  "mcpServers": {
    "kinship-earth": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/FULL/PATH/TO/kinship-earth-mcp",
        "--package", "kinship-orchestrator",
        "python", "-m", "kinship_orchestrator.server"
      ]
    }
  }
}
```

> Replace `/FULL/PATH/TO/kinship-earth-mcp` with the actual path where you cloned the repo.

### 3. Restart Claude Desktop

Restart Claude Desktop. You should see "kinship-earth" in the MCP server list (click the hammer icon).

### 4. Try it

Ask Claude:

> "What ecological monitoring sites exist near Portland, Oregon?"

> "Describe the available ecological data sources"

> "What was the climate like at latitude 45.5, longitude -122.7 on June 15, 2023?"

---

## What You Can Ask

### Cross-source queries (the unique value)

These combine multiple data sources in a single answer:

> "What were the environmental conditions when common dolphins were observed near Woods Hole, Massachusetts?"

> "Compare the climate at NEON's Wind River and Harvard Forest sites during summer 2023"

> "What ecological data exists for the Puget Sound region?"

### Species observations (OBIS — marine biodiversity)

> "Find blue whale sightings in the North Pacific from 2020-2023"

> "What marine species have been recorded near the Great Barrier Reef?"

> "Show me bottlenose dolphin occurrence data near Cape Cod"

### Climate data (ERA5 — global, 1940-present)

> "What was the temperature and precipitation at 47°N, 122°W during January 2024?"

> "Give me a week of hourly climate data for the Amazon rainforest"

> "What are the soil temperature and moisture conditions at this location?"

### Ecological monitoring sites (NEON — 81 US sites)

> "What NEON sites are in the Pacific Northwest?"

> "Tell me about the Wind River Experimental Forest site"

> "What data products does NEON collect?"

---

## Available MCP Tools

| Tool | What it does |
|------|-------------|
| `ecology_get_environmental_context` | Climate + nearest monitoring sites for any point and date |
| `ecology_search` | Unified search across all sources — species, sites, and climate |
| `ecology_whats_changed` | What arrived, left or shifted near a place in the last two weeks, compared with last year |
| `ecology_describe_sources` | What data sources are available and their capabilities |

You can also run individual data source servers for more granular access:

| Server | Tools | Data |
|--------|-------|------|
| `neonscience` | `neon_list_sites`, `neon_get_site`, `neon_list_data_products`, `neon_search_observations` | 81 US field sites, 180+ data products |
| `obis` | `obis_search_occurrences`, `obis_get_occurrence`, `obis_get_statistics` | 168M+ marine occurrence records |
| `era5` | `era5_get_climate`, `era5_get_daily_summary`, `era5_list_variables` | Global climate, hourly, 1940-present |

---

## Running Individual Servers

If you only need one data source, you can run individual servers in Claude Desktop:

```json
{
  "mcpServers": {
    "neon": {
      "command": "uv",
      "args": [
        "run", "--directory", "/FULL/PATH/TO/kinship-earth-mcp",
        "--package", "neonscience-mcp",
        "python", "-m", "neonscience_mcp.server"
      ]
    },
    "obis": {
      "command": "uv",
      "args": [
        "run", "--directory", "/FULL/PATH/TO/kinship-earth-mcp",
        "--package", "obis-mcp",
        "python", "-m", "obis_mcp.server"
      ]
    },
    "era5": {
      "command": "uv",
      "args": [
        "run", "--directory", "/FULL/PATH/TO/kinship-earth-mcp",
        "--package", "era5-mcp",
        "python", "-m", "era5_mcp.server"
      ]
    }
  }
}
```

Or use the unified launcher from the command line:

```bash
kinship-earth --list              # Show available servers
kinship-earth neonscience         # Start NEON server
kinship-earth orchestrator        # Start orchestrator (all sources)
```

---

## Data Sources

| Source | Coverage | Records | Quality | Auth |
|--------|----------|---------|---------|------|
| **NEON** | 81 US sites, 20 ecoclimatic domains | 180+ data products | Tier 1 (research-grade) | None |
| **OBIS** | Global oceans | 168M+ occurrences, 166K+ species | Tier 2 (community-validated) | None |
| **ERA5** | Global, 1940-present | Hourly at ~25km resolution | Tier 1 (calibrated reanalysis) | None |

All data includes full provenance: DOIs, citation strings, license info, and links back to the original source. Every record is traceable.

---

## Architecture

```
┌─────────────────────────────────────────┐
│  AI Agent (Claude Desktop, etc.)        │
└─────────────────┬───────────────────────┘
                  │ MCP (stdio)
┌─────────────────▼───────────────────────┐
│  Orchestrator                            │
│  ecology_get_environmental_context       │
│  ecology_search                          │
│  ecology_describe_sources                │
└──┬──────────────┬───────────────┬───────┘
   │              │               │
┌──▼───┐    ┌────▼────┐    ┌────▼────┐
│ NEON │    │  OBIS   │    │  ERA5   │
│ 81   │    │ 168M    │    │ Global  │
│sites │    │ marine  │    │ climate │
│      │    │ records │    │ 1940-   │
└──────┘    └─────────┘    └─────────┘
```

All data flows through a shared schema (`EcologicalObservation`) aligned with [Darwin Core](https://dwc.tdwg.org/terms/) standards. Results include relevance scoring, quality tiers, and full provenance.

**Monorepo structure:**
```
kinship-earth-mcp/
  shared/           ← EcologicalObservation schema + ranking layer
  servers/
    neonscience/    ← NEON adapter + MCP server
    obis/           ← OBIS adapter + MCP server
    era5/           ← ERA5 adapter + MCP server
    orchestrator/   ← Cross-source intelligence tools
  launcher/         ← Unified CLI
```

---

## Running Tests

```bash
uv run --package kinship-orchestrator pytest servers/ -v
```

36 tests across all servers, hitting real APIs. Tests verify connectivity, schema compliance, semantic correctness (is a dolphin actually in Cetacea?), and scientific fitness (does the data answer real research questions?).

---

## Roadmap

**Phase 1 (current) — MVP:** NEON + OBIS + ERA5 + Orchestrator. Done.

**Phase 2 — Expand:** Xeno-canto (bird audio), eBird (1.7B bird observations), SoilGrids (global soil). Source Registry for data discovery.

**Phase 3 — Intelligence:** PostgreSQL + pgvector for local caching, Knowledge Graph for ecological reasoning, cross-modal embeddings.

---

## Citation

If you use Kinship Earth in research, please cite the underlying data sources:

- **NEON:** National Ecological Observatory Network. https://www.neonscience.org
- **OBIS:** Ocean Biodiversity Information System. https://obis.org
- **ERA5:** Hersbach et al. (2020). The ERA5 global reanalysis. Q.J.R. Meteorol. Soc., 146(730), 1999-2049. DOI: [10.1002/qj.3803](https://doi.org/10.1002/qj.3803)

---

## License

MIT
