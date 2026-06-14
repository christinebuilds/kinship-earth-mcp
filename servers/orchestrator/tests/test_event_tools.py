"""Tests for ecology_events and ecology_subscribe tool registration."""

from __future__ import annotations

import pytest

from kinship_orchestrator.server import mcp


def test_ecology_events_tool_registered():
    tool_names = list(mcp._tool_manager._tools.keys())
    assert "ecology_events" in tool_names, f"ecology_events not found. Tools: {tool_names}"


def test_ecology_subscribe_tool_registered():
    tool_names = list(mcp._tool_manager._tools.keys())
    assert "ecology_subscribe" in tool_names, f"ecology_subscribe not found. Tools: {tool_names}"


def test_tool_count():
    tool_count = len(mcp._tool_manager._tools)
    assert tool_count == 22, f"Expected 22 tools, got {tool_count}"
