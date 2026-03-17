# Kinship Earth — Jupyter Notebook Examples

Example notebooks showing how to query ecological data from a Kinship Earth MCP server using Python.

## Notebooks

### [getting-started.ipynb](./getting-started.ipynb)
**Querying Ecological Data with Kinship Earth**

A single-site walkthrough covering:
- Connecting to the MCP server (stdio and HTTP options)
- Searching for species observations near Monterey Bay
- Converting results to pandas DataFrames
- Visualizing species distributions and climate trends
- Exporting results as GeoJSON for GIS tools

### [ecological-monitoring.ipynb](./ecological-monitoring.ipynb)
**Multi-Site Ecological Monitoring Dashboard**

A multi-site comparison workflow covering:
- Querying five ecosystems (Monterey Bay, Yellowstone, Everglades, Great Barrier Reef, Amazon)
- Building a comparison DataFrame with biodiversity and climate metrics
- Visualizing species richness and temperature across sites
- Exporting all results as GeoJSON files for QGIS

## Prerequisites

- Python 3.10+
- A running Kinship Earth MCP server (see main repo README for setup)

Install Python dependencies:

```bash
pip install mcp httpx pandas matplotlib
```

## Connecting to the Server

Both notebooks show two connection options:

**Option A — stdio (local development)**

The MCP client launches the server as a subprocess. Update the path in the notebook to point to your `kinship-earth-mcp` installation:

```python
from mcp.client.stdio import StdioServerParameters, stdio_client

server_params = StdioServerParameters(
    command="uv",
    args=["run", "--directory", "/path/to/kinship-earth-mcp/servers/orchestrator",
          "python", "-m", "kinship_orchestrator.server"],
)
```

**Option B — HTTP (remote server)**

Connect to a Kinship Earth server running in HTTP mode:

```python
from mcp.client.streamable_http import streamablehttp_client

KINSHIP_URL = "http://localhost:8000/mcp"  # Or your deployment URL
```

## GeoJSON Output

The notebooks demonstrate Kinship Earth's GeoJSON output format, which produces standard [RFC 7946](https://datatracker.ietf.org/doc/html/rfc7946) FeatureCollections. These files can be imported directly into:

- **QGIS** — Layer > Add Layer > Add Vector Layer
- **ArcGIS** — Add Data > select .geojson file
- **Kepler.gl** — drag and drop the file
- **Leaflet / Mapbox** — load as a GeoJSON source

## Environment Variables (optional)

Some Kinship Earth data sources accept API keys for higher rate limits:

| Variable | Source | Required? |
|----------|--------|-----------|
| `NEON_API_TOKEN` | NEON Science | Optional (higher rate limits) |
| `EBIRD_API_KEY` | eBird / Cornell Lab | Required for eBird tools |
