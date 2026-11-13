"""File problems from chats as GitHub issues, so the runner can fix and close them.

Two ways in:

- ``report_problem`` — called by the ``ecology_report_problem`` tool when a
  person (or Claude) in a chat says a result is wrong or something broke.
- ``watch_tools`` — wraps every MCP tool so an exception, or a result with an
  ``error`` key, is reported automatically.

Issues are public (the repo is public), so reports never carry chat text
beyond the summary Claude writes, and coordinates are rounded to about 1 km.
One open issue per distinct problem: a repeat adds at most one comment a day.

Needs ``GITHUB_TOKEN`` (fine-grained: Issues read/write on this repo only).
Without it, reporting is a silent no-op so the server works as before.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import os
import re
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

REPO = os.environ.get("KINSHIP_ISSUES_REPO", "christinebuilds/kinship-earth-mcp")
API = "https://api.github.com"
CHAT_LABELS = ["bug", "from-chat"]
AUTO_LABELS = ["bug", "auto-reported"]
_REPEAT_COMMENT_SECONDS = 24 * 3600
_last_comment: dict[str, float] = {}
_COORD = re.compile(r"(-?\d{1,3}\.\d{3,})")


def _token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or None


def coarsen(text: str) -> str:
    """Round long decimal numbers (coordinates) to two places, about 1 km."""
    return _COORD.sub(lambda m: f"{float(m.group(1)):.2f}", text or "")


def signature(*parts: str) -> str:
    """Stable id for 'the same problem': digits and quoted values are ignored."""
    norm = " | ".join(re.sub(r"'[^']*'|\"[^\"]*\"|\d+", "#", p or "") for p in parts)
    return hashlib.sha1(norm.encode()).hexdigest()[:12]


def _marker(sig: str) -> str:
    return f"<!-- kinship-problem:{sig} -->"


async def _request(method: str, path: str, **kw) -> httpx.Response:
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        return await client.request(method, API + path, headers=headers, **kw)


async def _find_open(sig: str, labels: list[str]) -> dict | None:
    r = await _request("GET", f"/repos/{REPO}/issues",
                       params={"state": "open", "labels": labels[-1], "per_page": 100})
    if r.status_code != 200:
        return None
    for issue in r.json():
        if _marker(sig) in (issue.get("body") or ""):
            return issue
    return None


async def file_issue(title: str, body: str, sig: str, labels: list[str]) -> dict:
    """Create the issue, or note a repeat on the open one. Never raises."""
    if not _token():
        return {"status": "not_configured"}
    try:
        existing = await _find_open(sig, labels)
        if existing:
            now = time.time()
            if now - _last_comment.get(sig, 0) > _REPEAT_COMMENT_SECONDS:
                _last_comment[sig] = now
                await _request("POST", f"/repos/{REPO}/issues/{existing['number']}/comments",
                               json={"body": "Seen again."})
            return {"status": "already_open", "issue": existing["number"], "url": existing.get("html_url")}
        payload = {"title": title[:200], "body": coarsen(body) + "\n\n" + _marker(sig), "labels": labels}
        r = await _request("POST", f"/repos/{REPO}/issues", json=payload)
        if r.status_code == 422:  # labels this token may not create
            payload.pop("labels")
            r = await _request("POST", f"/repos/{REPO}/issues", json=payload)
        if r.status_code >= 300:
            logger.warning("Problem report not filed: HTTP %s", r.status_code)
            return {"status": "failed", "http_status": r.status_code}
        issue = r.json()
        return {"status": "filed", "issue": issue["number"], "url": issue.get("html_url")}
    except Exception as e:  # reporting must never break a tool
        logger.warning("Problem report not filed: %s", e)
        return {"status": "failed"}


async def report_problem(summary: str, what_happened: str, expected: str = "", tool_name: str = "") -> dict:
    summary = coarsen(summary.strip())
    body = (
        "Reported from a Kinship Earth chat.\n\n"
        f"**What happened:** {coarsen(what_happened.strip())}\n\n"
        + (f"**Expected:** {coarsen(expected.strip())}\n\n" if expected.strip() else "")
        + (f"**Tool:** `{tool_name}`\n" if tool_name else "")
    )
    return await file_issue(f"[chat] {summary}", body, signature("chat", tool_name, summary), CHAT_LABELS)


def _report_in_background(tool: str, kind: str, detail: str) -> None:
    detail = coarsen(detail)[:500]
    title = f"[auto] {tool}: {detail.splitlines()[0][:120] if detail else kind}"
    body = f"`{tool}` returned an error during a chat.\n\n**{kind}:** {detail}\n"
    coro = file_issue(title, body, signature("auto", tool, kind, detail), AUTO_LABELS)
    try:
        asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        coro.close()


def _wrap(name: str, fn):
    @functools.wraps(fn)
    async def wrapped(*args, **kwargs):
        try:
            result = await fn(*args, **kwargs)
        except Exception as e:
            _report_in_background(name, type(e).__name__, str(e))
            raise
        if isinstance(result, dict) and result.get("error"):
            _report_in_background(name, "error result", str(result["error"]))
        return result
    return wrapped


def watch_tools(mcp, skip: tuple[str, ...] = ("ecology_report_problem",)) -> int:
    """Wrap every async tool on a FastMCP server. Returns how many were wrapped."""
    n = 0
    for name, tool in mcp._tool_manager._tools.items():
        if name in skip or not tool.is_async:
            continue
        tool.fn = _wrap(name, tool.fn)
        n += 1
    return n
