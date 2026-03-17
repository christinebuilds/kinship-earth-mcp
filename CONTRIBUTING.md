# Contributing to Kinship Earth

Thank you for your interest in contributing to Kinship Earth! This project makes ecological data from multiple scientific databases queryable through a unified interface.

## How to Contribute

### Adding a New Data Source Adapter

This is the most valuable contribution you can make. Each adapter connects a new ecological data source to the Kinship Earth schema.

**What you need:**
1. A data source with a public API (REST, GraphQL, or bulk download)
2. Python 3.12+
3. Familiarity with `httpx` for async HTTP and `pydantic` for data models

**Steps:**

```bash
# 1. Fork and clone
git clone https://github.com/YOUR-USERNAME/kinship-earth-mcp.git
cd kinship-earth-mcp

# 2. Create your adapter directory
mkdir -p servers/YOUR-SOURCE/src/your_source_mcp servers/YOUR-SOURCE/tests

# 3. Copy the structure from an existing adapter
#    Good templates: servers/gbif/ (simple) or servers/obis/ (with geo search)

# 4. Implement the adapter interface
#    - adapter.py: search(), get_by_id(), capabilities()
#    - server.py: MCP tool wrappers
#    - pyproject.toml: package config

# 5. Write tests (4 layers)
#    - Connectivity: does the API respond?
#    - Contract: does every observation have the required fields?
#    - Semantic: is the data factually correct?
#    - Scientific fitness: is the data usable for research?

# 6. Add to workspace
#    Edit pyproject.toml: add your adapter to the workspace members list

# 7. Run tests
uv sync
uv run --package your-source-mcp pytest servers/YOUR-SOURCE/tests/ -v
```

### The EcologicalObservation Schema

Every adapter must normalize its data into `EcologicalObservation` (defined in `shared/src/kinship_shared/schema.py`). Key fields:

- `id`: `"{source}:{source_id}"` format
- `modality`: occurrence, acoustic, sensor, chemical, etc.
- `taxon`: scientific name, common name, GBIF ID
- `location`: lat/lng (Darwin Core field names)
- `observed_at`: ISO 8601 datetime
- `quality`: tier (1-4), grade, confidence, flags
- `provenance`: source_api, DOI, license, citation_string

**Provenance is mandatory.** Every observation must include enough information to cite the data source properly.

### Reporting Bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Which adapter/tool was involved

### Improving Tests

More tests are always welcome. Our test philosophy:
- **Known-answer tests**: verify against ground truth (GPS coordinates, species taxonomy)
- **Multi-layer**: connectivity → contract → semantic → scientific fitness
- **Scientist personas**: simulate real research workflows

### Code Style

- Python 3.12+, type hints encouraged
- `uv` for package management
- Async/await for all API calls
- Darwin Core field names where applicable

## What We're Looking For

Priority data sources for new adapters:
- **Movebank** — animal GPS/movement tracking (6B records)
- **FLUXNET** — carbon/water/energy flux tower network
- **Wildlife Insights** — camera trap images (30M+ photos)
- **NOAA CO-OPS** — tide and water level stations
- **Copernicus Land** — satellite land cover data

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

Be kind, be constructive, respect the data and the ecosystems it represents.
