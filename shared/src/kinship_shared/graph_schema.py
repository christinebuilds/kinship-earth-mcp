"""
Knowledge graph entity and relationship models.

Defines the ontology for the ecological knowledge graph:
entities (species, locations, sites, queries) and relationships
(OBSERVED_AT, INHABITS, CORRELATES_WITH, etc.).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


EntityType = Literal[
    "species",
    "location",
    "watershed",
    "neon_site",
    "researcher",
    "query",
    "finding",
    "data_source",
    "pattern",
]

RelationshipType = Literal[
    "OBSERVED_AT",
    "INHABITS",
    "DRAINS_TO",
    "QUERIED_BY",
    "QUERIED_ABOUT",
    "FOUND_IN",
    "CORRELATES_WITH",
    "MONITORED_BY",
    "SOURCED_FROM",
    "SIMILAR_TO",
    # Canon evidential edges (docs/CANON.md)
    "DERIVES_FROM",
    "GENERALIZES",
    "SPECIAL_CASE_OF",
    "SUPPORTS",
    "CONTRADICTS",
    "REQUIRES",
    "ANALOG_OF",
    "TESTED_BY",
    "EVIDENCED_BY",
    "INFORMS",
]

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
    "INFORMS",
]


class GraphEntity(BaseModel):
    """A node in the ecological knowledge graph."""

    id: str = Field(description="e.g. 'species:delphinus_delphis' or 'location:41.50_-70.70'")
    entity_type: EntityType
    name: str
    properties: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    mention_count: int = 1


class GraphRelationship(BaseModel):
    """An edge in the ecological knowledge graph."""

    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: dict = Field(default_factory=dict)
    weight: float = 1.0
    first_seen: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    evidence_count: int = 1


class TemporalFact(BaseModel):
    """A fact with a validity window — can be superseded but not deleted."""

    id: str
    entity_id: str
    fact_type: str = Field(description="e.g. 'temperature_normal', 'species_range', 'population_estimate'")
    value: dict
    valid_from: datetime
    valid_until: Optional[datetime] = None  # None = still current
    source: str
    superseded_by: Optional[str] = None


# ---------------------------------------------------------------------------
# Entity ID helpers
# ---------------------------------------------------------------------------

def make_species_id(scientific_name: str) -> str:
    """Deterministic ID for a species entity."""
    return f"species:{scientific_name.lower().replace(' ', '_')}"


def make_location_id(lat: float, lng: float) -> str:
    """Deterministic ID for a location entity (~1km precision)."""
    return f"location:{lat:.2f}_{lng:.2f}"


def make_neon_site_id(site_code: str) -> str:
    """Deterministic ID for a NEON site entity."""
    return f"neon:{site_code}"


def make_query_id(turn_id: str) -> str:
    """ID for a query entity, linked to a conversation turn."""
    return f"query:{turn_id}"


def make_user_id(user_id: str) -> str:
    """ID for a researcher/user entity."""
    return f"user:{user_id}"


def make_source_id(adapter_id: str) -> str:
    """ID for a data source entity."""
    return f"source:{adapter_id}"


def make_pattern_id(slug: str) -> str:
    """ID for a canon pattern entity (slug is the canon node id)."""
    return f"pattern:{slug}"
