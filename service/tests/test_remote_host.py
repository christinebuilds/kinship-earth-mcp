"""The hosted /mcp endpoint must accept requests addressed to its public host."""

from fastapi.testclient import TestClient

from kinship_service.app import app

INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
}


def test_mcp_accepts_public_host():
    with TestClient(app, base_url="https://kinship-api-staging-40ec.up.railway.app") as client:
        r = client.post(
            "/mcp", json=INIT,
            headers={"Accept": "application/json, text/event-stream"},
        )
    assert r.status_code == 200, r.text
    assert "kinship-earth" in r.text
