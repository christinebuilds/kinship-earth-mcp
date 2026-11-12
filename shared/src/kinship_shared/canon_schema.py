"""
Kinship Canon node schema.

One node = one known ecological pattern, stored as Markdown with YAML
front matter in ``canon/patterns/<slug>.md``. See docs/CANON.md.

Cross-node rules (edge targets must resolve) live in ``canon_loader``;
everything that can be checked on a single node is validated here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .graph_schema import CanonEdgeType

NodeKind = Literal[
    "empirical_law", "theory", "model", "conjecture", "open_problem", "rule_of_thumb"
]
NodeStatus = Literal[
    "conjecture", "supported", "contested", "refuted_in_regime", "refuted"
]
NodeLayer = Literal[
    "organism", "population", "community", "ecosystem", "macro", "cross_field"
]
# Same values as Provenance.care_status in schema.py
CareStatus = Literal["public", "research", "restricted", "sovereign"]


class Variable(BaseModel):
    symbol: str
    meaning: str
    unit: str
    typical: Optional[str] = None
    source: Optional[str] = None


class Regime(BaseModel):
    """Where the pattern is claimed to hold."""

    taxa: list[str] = Field(default_factory=list)
    scales: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    type: Literal["paper", "dataset", "check_result", "expert"]
    doi: Optional[str] = None
    ref: Optional[str] = None
    source: Optional[str] = None
    product: Optional[str] = None
    role: Literal["origin", "support", "contradiction", "check", "cross_field_review"]
    note: Optional[str] = None


class Check(BaseModel):
    """An executable test of the pattern against federation data."""

    id: str
    runner: str = Field(description="Path to the check runner, e.g. canon/checks/foo.py")
    data: dict = Field(default_factory=dict)
    expect: dict = Field(default_factory=dict)
    last_run: Optional[datetime] = None
    result: Optional[dict] = None

    @property
    def passed(self) -> bool:
        """A check passes when its result records ``pass: true`` (or ``passed``)."""
        if not self.result:
            return False
        return self.result.get("pass") is True or self.result.get("passed") is True


class Edge(BaseModel):
    to: str
    type: CanonEdgeType
    field: Optional[str] = None
    note: Optional[str] = None
    confidence: Literal["established", "conjectural"] = "established"


class CanonNode(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    name: str
    aliases: list[str] = Field(default_factory=list)
    kind: NodeKind
    status: NodeStatus
    layer: NodeLayer
    statement_plain: str
    statement_formal: Optional[str] = None
    variables: list[Variable] = Field(default_factory=list)
    regime: Regime = Field(default_factory=Regime)
    known_exceptions: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    contributors: list[str] = Field(default_factory=list)
    care_status: CareStatus
    narrative: Optional[str] = Field(
        default=None, description="Markdown body below the front matter."
    )

    @model_validator(mode="after")
    def _evidence_rules(self) -> "CanonNode":
        if self.status == "supported":
            has_passing_check = any(c.passed for c in self.checks)
            n_support = sum(1 for e in self.evidence if e.role in ("support", "origin"))
            if not has_passing_check and n_support < 2:
                raise ValueError(
                    "status 'supported' requires a passing check or at least two "
                    "evidence entries with role 'support' or 'origin'"
                )
        if self.kind in ("conjecture", "open_problem"):
            if len(self.evidence) < 3:
                raise ValueError(
                    f"kind '{self.kind}' requires at least three evidence entries "
                    "(the what's-been-tried list)"
                )
            if self.checks:
                raise ValueError(f"kind '{self.kind}' must not carry checks")
        return self
