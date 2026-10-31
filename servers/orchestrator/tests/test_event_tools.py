"""Tests for ecology_events and ecology_subscribe tool registration (spec 013)."""

import pytest


def test_ecology_events_tool_registered():
    from kinship_orchestrator.server import mcp
    tools = list(mcp._tool_manager._tools.keys())
    assert "ecology_events" in tools, f"ecology_events not found in tools: {tools}"


def test_ecology_subscribe_tool_registered():
    from kinship_orchestrator.server import mcp
    tools = list(mcp._tool_manager._tools.keys())
    assert "ecology_subscribe" in tools, f"ecology_subscribe not found in tools: {tools}"
