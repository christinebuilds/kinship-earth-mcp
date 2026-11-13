"""Problem reports from chats become GitHub issues (offline: GitHub is faked)."""

import asyncio

import pytest

from kinship_orchestrator import problem_reports as pr


class FakeResponse:
    def __init__(self, status, data):
        self.status_code = status
        self._data = data

    def json(self):
        return self._data


@pytest.fixture
def github(monkeypatch):
    """A fake GitHub holding issues in memory; records every call."""
    state = {"issues": [], "calls": []}

    async def fake_request(method, path, **kw):
        state["calls"].append((method, path, kw))
        if method == "GET" and path.endswith("/issues"):
            return FakeResponse(200, [i for i in state["issues"] if i["state"] == "open"])
        if method == "POST" and path.endswith("/issues"):
            issue = {"number": len(state["issues"]) + 1, "state": "open", "html_url": "u", **kw["json"]}
            state["issues"].append(issue)
            return FakeResponse(201, issue)
        if method == "POST" and path.endswith("/comments"):
            return FakeResponse(201, {})
        return FakeResponse(404, {})

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setattr(pr, "_request", fake_request)
    pr._last_comment.clear()
    return state


def test_no_token_is_a_silent_noop(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = asyncio.run(pr.report_problem("x", "y"))
    assert result == {"status": "not_configured"}


def test_chat_report_files_issue_with_coarse_coordinates(github):
    result = asyncio.run(pr.report_problem(
        "no birds near 37.77251, -122.39348", "empty list at 37.77251,-122.39348", tool_name="ecology_whats_around_me"))
    assert result["status"] == "filed"
    issue = github["issues"][0]
    assert "37.77251" not in issue["title"] + issue["body"]
    assert "37.77" in issue["body"] and "-122.39" in issue["body"]
    assert issue["labels"] == ["bug", "from-chat"]


def test_repeat_report_comments_once_instead_of_new_issue(github):
    asyncio.run(pr.report_problem("same thing", "a"))
    second = asyncio.run(pr.report_problem("same thing", "b"))
    third = asyncio.run(pr.report_problem("same thing", "c"))
    assert len(github["issues"]) == 1
    assert second["status"] == third["status"] == "already_open"
    comments = [c for c in github["calls"] if c[1].endswith("/comments")]
    assert len(comments) == 1


def test_signature_ignores_numbers_and_quoted_values():
    assert pr.signature("t", "Turn 'abc' not found at 12") == pr.signature("t", "Turn 'xyz' not found at 99")


def test_watched_tool_reports_exception_and_still_raises(github):
    async def boom(lat: float):
        raise ValueError("upstream timed out after 15s")

    wrapped = pr._wrap("ecology_search", boom)

    async def run():
        with pytest.raises(ValueError):
            await wrapped(lat=1.0)
        await asyncio.sleep(0)  # let the background report run
        await asyncio.sleep(0)

    asyncio.run(run())
    assert github["issues"] and github["issues"][0]["labels"] == ["bug", "auto-reported"]


def test_server_wraps_tools_and_exposes_report_tool():
    from kinship_orchestrator.server import mcp

    tools = mcp._tool_manager._tools
    assert "ecology_report_problem" in tools
    assert tools["ecology_search"].fn.__wrapped__ is not None
