"""Spec 021: canon CI findings (offline)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("canon_ci", ROOT / "tools" / "canon_ci.py")
canon_ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(canon_ci)


def node(id_, name, aliases="[]", status="conjecture", extra=""):
    return f"""---
id: {id_}
name: {name}
aliases: {aliases}
kind: empirical_law
status: {status}
layer: population
statement_plain: Something holds.
care_status: public
{extra}---
"""


def make(tmp_path, files: dict, stubs=None, runners=()):
    pats = tmp_path / "canon" / "patterns"
    pats.mkdir(parents=True)
    for name, text in files.items():
        (pats / name).write_text(text)
    if stubs is not None:
        (pats / "_stubs.yaml").write_text("".join(f"- {s}\n" for s in stubs))
    for r in runners:
        p = tmp_path / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# runner\n")
    return pats


def test_real_canon_passes():
    assert canon_ci.ci_findings(ROOT / "canon" / "patterns") == []


def test_duplicate_alias(tmp_path):
    pats = make(tmp_path, {"a.md": node("a", "A law", "[shared name]"),
                           "b.md": node("b", "B law", "[shared name]")})
    out = canon_ci.ci_findings(pats)
    assert any("shared name" in f and "a, b" in f for f in out)


def test_stub_rules(tmp_path):
    pats = make(tmp_path, {"a.md": node("a", "A law")}, stubs=["a", "orphan"])
    out = canon_ci.ci_findings(pats)
    assert any("'a' is now a written node" in f for f in out)
    assert any("'orphan' is not referenced" in f for f in out)


CHECK = """checks:
  - id: c1
    runner: canon/checks/c1.py
    data: {{adapter: {adapter}}}
{result}"""


def test_missing_runner(tmp_path):
    pats = make(tmp_path, {"a.md": node("a", "A law", extra=CHECK.format(adapter="neonscience", result=""))})
    assert any("runner not found" in f for f in canon_ci.ci_findings(pats))


def test_supported_with_failed_result(tmp_path):
    extra = ("evidence:\n  - {type: paper, ref: X, role: origin}\n  - {type: paper, ref: Y, role: support}\n"
             + CHECK.format(adapter="neonscience", result="    result: {pass: false}\n"))
    pats = make(tmp_path, {"a.md": node("a", "A law", status="supported", extra=extra)},
                runners=["canon/checks/c1.py"])
    assert any("last failed" in f for f in canon_ci.ci_findings(pats))


def test_presence_only_guard(tmp_path):
    extra = ("regime:\n  conditions: [\"requires true counts\"]\n"
             + CHECK.format(adapter="gbif", result=""))
    pats = make(tmp_path, {"a.md": node("a", "A law", extra=extra)}, runners=["canon/checks/c1.py"])
    assert any("presence-only" in f for f in canon_ci.ci_findings(pats))
