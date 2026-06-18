"""
Unified launcher for Kinship Earth MCP servers.

Usage:
  kinship-earth <server-name> [transport]

Servers:
  neonscience   NEON ecological observatory data (81 US sites)
  obis          Ocean Biodiversity Information System (168M+ records)
  era5          ERA5 climate reanalysis (global, 1940-present)
  orchestrator  Cross-source ecological intelligence tools

Transport:
  stdio         Standard I/O (default, for MCP clients)
  sse           Server-Sent Events (for HTTP access)

Examples:
  kinship-earth neonscience          # Start NEON server via stdio
  kinship-earth orchestrator sse     # Start orchestrator via HTTP
  kinship-earth --list               # List available servers
"""

import sys


SERVERS = {
    "neonscience": {
        "module": "neonscience_mcp.server",
        "description": "NEON — 81 US ecological observatory sites, 180+ data products",
    },
    "obis": {
        "module": "obis_mcp.server",
        "description": "OBIS — 168M+ marine species occurrence records worldwide",
    },
    "era5": {
        "module": "era5_mcp.server",
        "description": "ERA5 — Global climate reanalysis, hourly data from 1940",
    },
    "orchestrator": {
        "module": "kinship_orchestrator.server",
        "description": "Cross-source tools: environmental context, unified search",
    },
    "movebank": {
        "module": "movebank_mcp.server",
        "description": "Movebank — animal tracking and GPS telemetry (6,000+ studies)",
    },
    "fluxnet": {
        "module": "fluxnet_mcp.server",
        "description": "FLUXNET / AmeriFlux — carbon flux tower measurements (~950 sites)",
    },
}


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        print(__doc__)
        sys.exit(0)

    if args[0] == "--list":
        print("Available Kinship Earth MCP servers:\n")
        for name, info in SERVERS.items():
            print(f"  {name:<16} {info['description']}")
        print(f"\nUsage: kinship-earth <server-name> [stdio|sse]")
        sys.exit(0)

    server_name = args[0]
    transport = args[1] if len(args) > 1 else "stdio"

    if server_name not in SERVERS:
        print(f"Unknown server: {server_name!r}")
        print(f"Available servers: {', '.join(SERVERS.keys())}")
        sys.exit(1)

    if transport not in ("stdio", "sse"):
        print(f"Unknown transport: {transport!r}. Use 'stdio' or 'sse'.")
        sys.exit(1)

    # Import and run the server
    module_name = SERVERS[server_name]["module"]
    module = __import__(module_name, fromlist=["mcp"])
    server = module.mcp
    server.run(transport=transport)


if __name__ == "__main__":
    main()
