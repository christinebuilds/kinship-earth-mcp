"""kinship-shared: unified schema and adapter interface for Kinship Earth."""

from .adapter import EcologicalAdapter
from .ecology_tools import (
    run_describe_sources,
    run_get_environmental_context,
    run_search,
)
from .geojson import observations_to_geojson
from .ranking import rank_observations, score_observation
from .retry import http_get_with_retry
from .schema import (
    AdapterCapabilities,
    EcologicalObservation,
    Location,
    Provenance,
    Quality,
    SearchParams,
    SearchRelevance,
    SignalModality,
    TaxonInfo,
)

__all__ = [
    "EcologicalAdapter",
    "EcologicalObservation",
    "AdapterCapabilities",
    "SearchParams",
    "SearchRelevance",
    "SignalModality",
    "Location",
    "TaxonInfo",
    "Quality",
    "Provenance",
    "rank_observations",
    "score_observation",
    "http_get_with_retry",
    "run_describe_sources",
    "run_get_environmental_context",
    "run_search",
    "observations_to_geojson",
]
