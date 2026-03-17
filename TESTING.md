# Kinship Earth — Tester Guide

Thanks for testing! You'll be using Claude Desktop to query ecological data through Kinship Earth. The whole setup takes about 5 minutes.

---

## Prerequisites

- **Claude Desktop** — [download here](https://claude.ai/download) (free account works)
- **Python 3.12+** — check with `python3 --version`
- **uv** (Python package manager) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Setup (5 minutes)

### 1. Clone and install

```bash
git clone https://github.com/christinebuilds/kinship-earth-mcp.git
cd kinship-earth-mcp
./setup-claude-desktop.sh
```

The script will install dependencies and print a JSON config block.

### 2. Add to Claude Desktop

1. Open Claude Desktop
2. Go to **Settings > Developer > Edit Config**
3. Paste the JSON config block from the setup script
4. Save and **restart Claude Desktop**

### 3. Verify it works

After restarting, click the **hammer icon** in Claude Desktop. You should see "kinship-earth" listed with 3 tools.

Ask Claude:

> "What ecological data sources are available?"

If you get a structured response listing NEON, OBIS, and ERA5 — you're set.

---

## What to try

Pick one or two of these scenarios. Each takes 2-3 minutes.

### Scenario 1: Forest ecology

Ask Claude these questions in order:

1. "What were the environmental conditions at Wind River Experimental Forest (latitude 45.82, longitude -121.95) during June 15-21, 2023?"
2. "What data products does the WREF NEON site collect?"
3. "Are there any marine biodiversity records within 200km of Wind River?"

### Scenario 2: Marine research

1. "Search for common dolphin (Delphinus delphis) sightings within 200km of Woods Hole, Massachusetts (latitude 41.5, longitude -70.7) from 2015 to 2023."
2. "Tell me more about the highest-ranked dolphin sighting."
3. "What ecological data sources are available, and what monitoring exists near this location?"

### Scenario 3: Open exploration

1. "What ecological data sources does Kinship Earth have access to?"
2. "What ecological data exists near Seattle, Washington (latitude 47.6, longitude -122.3)?"
3. "What was the climate like in this area during the record heat dome in late June 2021?"

### Scenario 4: Your own question

Try asking about a location or species you actually care about. This is the most valuable feedback.

---

## What I'd love to hear from you

After trying it, text/email me your thoughts on:

1. **Did it work?** Any errors during setup or when asking questions?
2. **Was the data useful?** Did the answers feel like real, trustworthy information?
3. **What's missing?** What did you wish it could do that it couldn't?
4. **Would you use this again?** For what?

Rough impressions are great — don't overthink it.

---

## Troubleshooting

**"kinship-earth" doesn't appear in Claude Desktop:**
- Make sure you restarted Claude Desktop after editing the config
- Check that the path in the config matches where you cloned the repo

**Tool calls fail or timeout:**
- The data comes from live APIs (NEON, OBIS, Open-Meteo). If one is down, that part of the response may be missing.
- Try again in a minute — it's usually transient.

**"uv: command not found":**
- Close and reopen your terminal after installing uv, or run `source ~/.bashrc`
