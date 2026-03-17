# QGIS Sample Data

Pre-built GeoJSON files for testing the Kinship Earth + QGIS integration without running any queries.

## Files

### monterey-bay-observations.geojson

A GeoJSON FeatureCollection containing 10 realistic marine species observations near Monterey Bay, California.

**Species included:**
- Humpback Whale (*Megaptera novaeangliae*)
- Southern Sea Otter (*Enhydra lutris nereis*)
- Northern Elephant Seal (*Mirounga angustirostris*)
- Gray Whale (*Eschrichtius robustus*)
- Leatherback Sea Turtle (*Dermochelys coriacea*)
- Ocean Sunfish (*Mola mola*)
- California Sea Lion (*Zalophus californianus*)
- Harbor Seal (*Phoca vitulina*)
- Common Bottlenose Dolphin (*Tursiops truncatus*)
- Pacific Sea Nettle (*Chrysaora fuscescens*)

**Coordinate area:** Monterey Bay (~36.55-36.80 N, ~121.79-122.02 W)

**Coordinate reference system:** WGS 84 (EPSG:4326)

## How to Load in QGIS

1. Open QGIS
2. Go to **Layer > Add Layer > Add Vector Layer** (or drag the file directly onto the map canvas)
3. In the dialog, click **Browse** and select `monterey-bay-observations.geojson`
4. Click **Add**, then **Close**
5. The observations will appear as points on the map

## Available Columns

Each feature has these properties you can use for styling, filtering, and labeling:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `scientific_name` | string | Binomial species name | Megaptera novaeangliae |
| `common_name` | string | Common English name | Humpback Whale |
| `observed_at` | string (ISO 8601) | Date and time of observation | 2025-09-15T14:32:00Z |
| `quality_grade` | string | Data quality level | research, verifiable |
| `source` | string | Data source | iNaturalist, OBIS |
| `observation_id` | string | Source-specific record ID | inat-198234567 |

## Styling Suggestions

- **Categorized by species**: Right-click layer > Properties > Symbology > Categorized > Column: `scientific_name`
- **Categorized by source**: Use `source` column to color iNaturalist vs OBIS records differently
- **Filtered by quality**: Right-click > Filter > `"quality_grade" = 'research'` to show only research-grade observations
- **Labeled**: Properties > Labels > Single Labels > Value: `common_name`
- **Temporal filter**: Use `observed_at` with QGIS Temporal Controller for time-based animation
