"""Spec 020: canon tools over a temporary canon directory (offline)."""

import os
import time
from pathlib import Path

import pytest

from kinship_shared import canon_tools as ct
from kinship_shared.canon_loader import load_node

SUPPORTED = """---
id: kleibers-law
name: Kleiber's law
aliases: [metabolic scaling, three-quarter power law]
kind: empirical_law
status: supported
layer: organism
statement_plain: Metabolic rate scales with body mass to about the 3/4 power.
evidence:
  - {type: paper, ref: "Kleiber 1932", role: origin}
checks:
  - id: c1
    runner: canon/checks/kleiber.py
    last_run: 2026-11-01T00:00:00Z
    result: {pass: true, b: 0.74}
edges:
  - {to: taylors-power-law, type: ANALOG_OF, field: physics}
care_status: public
---
Body.
"""

CONJECTURE = """---
id: taylors-power-law
name: Taylor's power law
aliases: [fluctuation scaling, TPL]
kind: empirical_law
status: conjecture
layer: population
statement_plain: Variance of counts grows as a power of the mean.
checks:
  - id: neon-landbird-tpl
    runner: canon/checks/does_not_exist.py
edges:
  - {to: tweedie-distributions, type: ANALOG_OF, field: statistics}
care_status: public
---
"""


@pytest.fixture
def canon(tmp_path: Path) -> Path:
    (tmp_path / "kleibers-law.md").write_text(SUPPORTED)
    (tmp_path / "taylors-power-law.md").write_text(CONJECTURE)
    (tmp_path / "_stubs.yaml").write_text("- tweedie-distributions\n")
    return tmp_path


def test_list_and_filter(canon):
    assert {n["id"] for n in ct.canon_list(canon_dir=canon)} == {"kleibers-law", "taylors-power-law"}
    only = ct.canon_list(status="conjecture", canon_dir=canon)
    assert [n["id"] for n in only] == ["taylors-power-law"]
    assert only[0]["status"] == "conjecture"


@pytest.mark.parametrize("q", ["taylors-power-law", "Taylor's power law", "TPL", "taylors law", "Taylor's law"])
def test_get_by_id_name_alias_loose(canon, q):
    assert ct.canon_get(q, canon_dir=canon)["id"] == "taylors-power-law"


def test_get_unknown(canon):
    assert ct.canon_get("species-area relationship", canon_dir=canon) is None


def test_neighbors_marks_stub_and_incoming(canon):
    nb = ct.canon_neighbors("taylors-power-law", canon_dir=canon)
    out = [e for e in nb["edges"] if e["direction"] == "out"]
    inc = [e for e in nb["edges"] if e["direction"] == "in"]
    assert out[0]["to"] == "tweedie-distributions" and out[0]["to_is_stub"] is True
    assert inc[0]["from"] == "kleibers-law"


def test_check_status_reads_only(canon):
    k = ct.canon_check_status("kleibers-law", canon_dir=canon)
    t = ct.canon_check_status("taylors-power-law", canon_dir=canon)
    assert k[0]["passed"] is True
    assert t[0]["passed"] is None and t[0]["result"] is None  # runner file absent; never run


def test_propose_valid_round_trips(canon, tmp_path):
    proposal = """
id: species-area-relationship
name: Species-area relationship
kind: empirical_law
status: conjecture
layer: community
statement_plain: Larger areas hold more species, roughly as a power of area.
care_status: public
"""
    res = ct.canon_propose(proposal, narrative="Arrhenius 1921.", canon_dir=canon)
    assert res["valid"] is True and res["duplicates"] == []
    f = tmp_path / "out" / "species-area-relationship.md"
    f.parent.mkdir()
    f.write_text(res["file_text"])
    assert load_node(f).id == "species-area-relationship"
    assert not (canon / "species-area-relationship.md").exists()


def test_propose_duplicate_alias(canon):
    proposal = """
id: fluctuation-scaling
name: Fluctuation scaling
aliases: [TPL]
kind: empirical_law
status: conjecture
layer: population
statement_plain: Same thing, new name.
care_status: public
"""
    res = ct.canon_propose(proposal, canon_dir=canon)
    assert res["valid"] is False and res["duplicates"] == ["taylors-power-law"]


def test_propose_supported_without_evidence_rejected(canon):
    proposal = """
id: new-law
name: New law
kind: empirical_law
status: supported
layer: population
statement_plain: Something.
care_status: public
"""
    res = ct.canon_propose(proposal, canon_dir=canon)
    assert res["valid"] is False
    assert "requires a passing check" in res["errors"][0]


def test_cache_reloads_after_edit(canon):
    assert ct.canon_get("TPL", canon_dir=canon)["status"] == "conjecture"
    p = canon / "taylors-power-law.md"
    p.write_text(p.read_text().replace("name: Taylor's power law", "name: Taylor's law (edited)"))
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert ct.canon_get("TPL", canon_dir=canon)["name"] == "Taylor's law (edited)"
