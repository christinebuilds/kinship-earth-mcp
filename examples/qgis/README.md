# QGIS + Kinship Earth MCP Integration Guide

Kinship Earth MCP provides ecological data (species observations, climate, monitoring sites) that can be visualized and analyzed in [QGIS](https://qgis.org/), the leading open-source GIS platform. This guide explains how an AI agent bridges the two systems using the Model Context Protocol.

## What You Can Do

- **Map species observations**: Query iNaturalist, OBIS, or eBird data and display occurrences as point layers in QGIS
- **Overlay climate data**: Fetch ERA5 temperature, precipitation, and wind data for any location and time period
- **Find monitoring infrastructure**: Locate NEON field sites near your study area and add them to your map
- **Spatial analysis**: Combine ecological data with QGIS processing tools (buffer zones, heatmaps, spatial joins)
- **Multi-source comparison**: Layer marine (OBIS), terrestrial (iNaturalist), and avian (eBird) data on the same map

## Prerequisites

### Software
1. **QGIS** (3.28+) — [download](https://qgis.org/download/)
2. **QGIS MCP Server** — [github.com/jjsantos01/qgis_mcp](https://github.com/jjsantos01/qgis_mcp)
   - Install the QGIS plugin (socket server component)
   - Install the MCP server (Python component)
3. **Kinship Earth MCP Server** — [github.com/christineyws-beep/kinship-earth-mcp](https://github.com/christineyws-beep/kinship-earth-mcp)
4. **An MCP-capable AI client** — Claude Desktop, Cursor, or any client that supports multiple MCP servers

### API Keys (Optional)
- `EBIRD_API_KEY` — required for eBird data ([request one](https://ebird.org/api/keygen))
- `NEON_API_TOKEN` — optional, increases NEON rate limits

## How It Works

The integration model uses an **AI agent as mediator** between two independent MCP servers:

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Kinship Earth   │◄───►│    AI Agent       │◄───►│   QGIS MCP    │
│  MCP Server      │     │  (Claude, etc.)   │     │   Server      │
│                  │     │                   │     │               │
│  ecology_search  │     │  Mediates between │     │  add_vector   │
│  era5_get_climate│     │  both servers     │     │  zoom_to_layer│
│  inaturalist_*   │     │                   │     │  execute_proc │
└─────────────────┘     └──────────────────┘     └───────┬───────┘
                                                          │ socket
                                                   ┌──────▼───────┐
                                                   │  QGIS Desktop │
                                                   └──────────────┘
```

There is **no direct connection** between Kinship Earth and QGIS. The AI agent:
1. Calls Kinship Earth tools to fetch ecological data
2. Converts the results to GeoJSON format
3. Writes the GeoJSON to a file
4. Calls QGIS MCP tools to load and display the file

## Claude Desktop Configuration

Add both servers to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kinship-earth": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/kinship-earth-mcp/servers/orchestrator",
        "run", "main.py"
      ],
      "env": {
        "EBIRD_API_KEY": "your-key-here"
      }
    },
    "qgis": {
      "command": "python",
      "args": ["/path/to/qgis_mcp/qgis_mcp_server.py"]
    }
  }
}
```

## Example Workflow: Marine Species Near Monterey Bay

### Step 1: Query ecological data

Ask the AI agent:

> "Search for marine species observations near Monterey Bay (36.6, -121.9) within 50km"

The agent calls `ecology_search` on Kinship Earth:

```
ecology_search(
  lat=36.6,
  lon=-121.9,
  radius_km=50,
  limit=50
)
```

### Step 2: Agent converts results to GeoJSON

The agent takes the species occurrence data and writes a GeoJSON FeatureCollection:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-121.87, 36.62]
      },
      "properties": {
        "scientific_name": "Megaptera novaeangliae",
        "common_name": "Humpback Whale",
        "observed_at": "2025-09-15",
        "quality_grade": "research",
        "source": "iNaturalist"
      }
    }
  ]
}
```

### Step 3: Agent loads data into QGIS

The agent calls QGIS MCP tools:

```
add_vector_layer(path="/tmp/monterey-bay-observations.geojson", name="Species Observations")
zoom_to_layer(layer_id="Species Observations")
```

### Step 4: Style and analyze in QGIS

Ask the agent to apply categorized styling:

> "Color the points by species. Use a graduated color ramp."

The agent can call `execute_processing` to run QGIS Processing Toolbox algorithms for spatial analysis (buffers, heatmaps, etc.).

<!-- Screenshot placeholder: QGIS map showing Monterey Bay with colored species observation points -->

## Alternative: Python Script (No AI Agent)

If you prefer a non-AI workflow, use the included `fetch_ecological_data.py` script:

```bash
python fetch_ecological_data.py --lat 36.6 --lon -121.9 --radius 50 --output monterey-bay.geojson
```

Then load the resulting GeoJSON in QGIS via **Layer > Add Layer > Add Vector Layer**.

## Sample Data

The `sample-data/` directory contains pre-built GeoJSON files you can load in QGIS immediately to test the workflow without running any queries:

- `monterey-bay-observations.geojson` — 10 marine species observations near Monterey Bay

See `sample-data/README.md` for loading instructions.

## Troubleshooting

### QGIS MCP server won't connect
- Make sure the QGIS plugin is installed and the socket server is running (check QGIS Python console)
- Default socket port is 9876. Ensure nothing else is using it.
- Restart QGIS after installing the plugin.

### GeoJSON won't load in QGIS
- Validate your GeoJSON at [geojson.io](https://geojson.io/) or with `ogr2ogr`
- Ensure coordinates are in `[longitude, latitude]` order (GeoJSON standard, NOT `[lat, lon]`)
- Check that the file has a `.geojson` extension

### No data returned from Kinship Earth
- Check that the MCP server is running (`uv run main.py` in the orchestrator directory)
- For eBird data, ensure `EBIRD_API_KEY` is set in environment
- Try a broader radius or different location — some areas have sparse data
- OBIS only has marine data; use iNaturalist for terrestrial species

### Coordinate system issues
- All Kinship Earth output uses WGS 84 (EPSG:4326) — the GeoJSON standard
- If overlaying with other layers, ensure they use the same CRS or reproject in QGIS

### AI agent confuses the two MCP servers
- Be explicit in your prompts: "Use Kinship Earth to search" and "Use QGIS to load the layer"
- If the agent calls the wrong tool, remind it which server provides data vs. which controls QGIS
