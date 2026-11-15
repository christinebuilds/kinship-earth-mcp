"""
Pydantic models for the Kinship Canon — one node file per known ecological pattern.

Node files live in canon/patterns/<slug>.md with YAML front matter.
See docs/CANON.md for the full format and contribution model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

CanonEdgeType = Literal[
    "DERIVES_FROM",
    "GENERALIZES",
    "SPECIAL_CASE_OF",
    "SUPPORTS",
    "CONTRADICTS",
    "REQUIRES",
    "ANALOG_OF",
    "TESTED_BY",
    "EVIDENCED_BY",
]


class Variable(BaseModel):
    symbol: str
    meaning: str
    unit: str
    typical: Optional[str] = None
    source: Optional[str] = None


class Regime(BaseModel):
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
    id: str
    runner: str
    data: dict = Field(default_factory=dict)
    expect: dict = Field(default_factory=dict)
    last_run: Optional[datetime] = None
    result: Optional[dict] = None


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
    kind: Literal[
        "empirical_law", "theory", "model",
        "conjecture", "open_problem", "rule_of_thumb",
    ]
    status: Literal[
        "conjecture", "supported", "contested",
        "refuted_in_regime", "refuted",
    ]
    layer: Literal[
        "organism", "population", "community",
        "ecosystem", "macro", "cross_field",
    ]
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
    care_status: Optional[Literal["public", "research", "restricted", "sovereign"]] = None
    narrative: Optional[str] = None  # body text below the YAML front matter

    @model_validator(mode="after")
    def _validate_status_and_kind(self) -> "CanonNode":
        if self.status == "supported":
            has_check_with_result = any(c.result is not None for c in self.checks)
            origin_support = [e for e in self.evidence if e.role in ("origin", "support")]
            if not has_check_with_result and len(origin_support) < 2:
                raise ValueError(
                    "status 'supported' requires either a check with a non-null result "
                    "or at least 2 evidence entries with role 'origin' or 'support'"
                )
        if self.kind in ("conjecture", "open_problem"):
            if len(self.evidence) < 3:
                raise ValueError(
                    f"kind '{self.kind}' requires at least 3 evidence entries "
                    "(the 'what's been tried' list)"
                )
            if self.checks:
                raise ValueError(
                    f"kind '{self.kind}' must have empty checks"
                )
        return self
