"""
EcologicalObservation schema — the unified data model for Kinship Earth.

Darwin Core field names are used throughout. Every piece of ecological data —
bird sighting, soil reading, frog call, plant stress signal — maps to this envelope.

Ref: https://dwc.tdwg.org/terms/
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Signal modality
# ---------------------------------------------------------------------------

SignalModality = Literal[
    "acoustic",       # sound recordings, bioacoustics
    "chemical",       # VOCs, soil chemistry, water chemistry
    "electrical",     # plant electrophysiology, bioelectricity
    "geospatial",     # satellite imagery, LiDAR
    "hydrological",   # stream flow, water chemistry, eDNA (Phase 1C)
    "movement",       # GPS telemetry, migration tracking (Phase 1C)
    "occurrence",     # species observation records
    "sensor",         # continuous IoT/instrument data (soil moisture, flux, climate)
    "spectral",       # hyperspectral, multispectral remote sensing (Phase 1C)
    "visual",         # camera traps, photos, video
]

QualityGrade = Literal["research", "community", "casual", "raw"]


# ---------------------------------------------------------------------------
# Component models
# ---------------------------------------------------------------------------

class TaxonInfo(BaseModel):
    """Taxonomic identity of the observed subject. Darwin Core-aligned."""

    scientific_name: str = Field(
        description="DwC: scientificName. Binomial or higher-rank name."
    )
    common_name: Optional[str] = Field(
        default=None,
        description="DwC: vernacularName."
    )
    gbif_id: Optional[int] = Field(
        default=None,
        description="DwC: taxonKey. Canonical GBIF backbone taxon ID — the cross-source entity reference."
    )
    rank: Optional[str] = Field(
        default=None,
        description="DwC: taxonRank. E.g. 'SPECIES', 'GENUS', 'FAMILY'."
    )
    kingdom: Optional[str] = Field(default=None, description="DwC: kingdom.")
    phylum: Optional[str] = Field(default=None, description="DwC: phylum.")
    class_: Optional[str] = Field(default=None, alias="class", description="DwC: class.")
    order: Optional[str] = Field(default=None, description="DwC: order.")
    family: Optional[str] = Field(default=None, description="DwC: family.")

    model_config = {"populate_by_name": True}


class Location(BaseModel):
    """Spatial context of the observation. Darwin Core-aligned."""

    lat: float = Field(description="DwC: decimalLatitude.")
    lng: float = Field(description="DwC: decimalLongitude.")
    elevation_m: Optional[float] = Field(
        default=None,
        description="DwC: verbatimElevation. Elevation in metres above sea level."
    )
    site_id: Optional[str] = Field(
        default=None,
        description="DwC: locationID. E.g. NEON site code 'WREF'."
    )
    site_name: Optional[str] = Field(
        default=None,
        description="DwC: locality. Human-readable place name."
    )
    uncertainty_m: Optional[float] = Field(
        default=None,
        description="DwC: coordinateUncertaintyInMeters."
    )
    country: Optional[str] = Field(
        default=None,
        description="DwC: country. Full country name."
    )
    country_code: Optional[str] = Field(
        default=None,
        description="DwC: countryCode. ISO 3166-1 alpha-2."
    )
    state_province: Optional[str] = Field(
        default=None,
        description="DwC: stateProvince."
    )
    watershed_id: Optional[str] = Field(
        default=None,
        description="Watershed identifier, e.g. USGS HUC-12 code. Enables watershed-level queries."
    )
    ecosystem_id: Optional[str] = Field(
        default=None,
        description="Ecosystem identifier from a recognized classification (e.g. EPA Level III Ecoregion)."
    )


class Quality(BaseModel):
    """Data quality and validation metadata."""

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Model or human confidence score (0–1)."
    )
    validated: Optional[bool] = Field(
        default=None,
        description="Whether the observation has been validated by a reviewer."
    )
    grade: Optional[QualityGrade] = Field(
        default=None,
        description="Quality grade: research > community > casual > raw."
    )
    tier: Optional[int] = Field(
        default=None,
        ge=1, le=4,
        description=(
            "Data quality tier: "
            "1=calibrated instrument (NEON/FLUXNET), "
            "2=community-validated (eBird research grade), "
            "3=citizen science (iNaturalist casual), "
            "4=raw/unvalidated (personal sensors)."
        )
    )
    flags: Optional[list[str]] = Field(
        default=None,
        description="DwC-style data quality flags. Exposed, never hidden."
    )


class Provenance(BaseModel):
    """Full provenance for data citation and traceability."""

    source_api: str = Field(
        description="Short ID of the source API: 'neonscience', 'gbif', 'ebird', etc."
    )
    source_id: str = Field(
        description="Original record ID in the source system."
    )
    original_url: Optional[str] = Field(
        default=None,
        description="Direct URL to the record in the source system."
    )
    doi: Optional[str] = Field(
        default=None,
        description="Dataset or record DOI."
    )
    license: Optional[str] = Field(
        default=None,
        description="SPDX license identifier or URL. E.g. 'CC-BY-4.0'."
    )
    attribution: Optional[str] = Field(
        default=None,
        description="Attribution string required by the license."
    )
    citation_string: Optional[str] = Field(
        default=None,
        description="Full citation string as required by the source. NEON and GBIF have specific formats."
    )
    dataset_id: Optional[str] = Field(
        default=None,
        description="DwC: datasetID. Identifier of the source dataset."
    )
    institution_code: Optional[str] = Field(
        default=None,
        description="DwC: institutionCode."
    )
    collection_code: Optional[str] = Field(
        default=None,
        description="DwC: collectionCode."
    )
    care_status: Optional[Literal["public", "research", "restricted", "sovereign"]] = Field(
        default=None,
        description=(
            "CARE Principles data governance status. "
            "'sovereign' = Indigenous data requiring community consent for use. "
            "Built into schema from day one per Kinship Earth design principles."
        )
    )
    sensor_id: Optional[str] = Field(
        default=None,
        description="Identifier of the specific sensor or instrument that produced this data."
    )
    collection_method: Optional[str] = Field(
        default=None,
        description="Method of data collection. E.g. 'autonomous_recorder', 'visual_survey', 'satellite_remote_sensing'."
    )


# ---------------------------------------------------------------------------
# The unified observation model
# ---------------------------------------------------------------------------

class EcologicalObservation(BaseModel):
    """
    The unified data model for Kinship Earth.

    Every piece of ecological data — bird sighting, soil reading, frog call,
    plant stress signal — maps to this envelope. The schema is the protocol.
    """

    # Identity
    id: str = Field(
        description=(
            "Globally unique ID across all sources. "
            "Format: '{source_api}:{source_id}'. "
            "E.g. 'neonscience:WREF:DP1.10003:2026-02', 'gbif:1234567890'."
        )
    )
    modality: SignalModality

    # Subject (optional for pure sensor readings with no taxonomic subject)
    taxon: Optional[TaxonInfo] = None

    # Spatiotemporal (always required)
    location: Location
    observed_at: datetime = Field(description="DwC: eventDate. ISO 8601.")
    duration_seconds: Optional[float] = Field(
        default=None,
        description="DwC: samplingEffort. Duration of the observation in seconds."
    )

    # The actual data payload (modality-specific)
    value: Optional[Any] = Field(
        default=None,
        description=(
            "Modality-specific payload. "
            "occurrence: {'count': 3, 'behavior': 'singing'}. "
            "sensor: {'soil_moisture_vwc': 0.28, 'depth_cm': 15}. "
            "acoustic: {'species_code': 'amro', 'confidence': 0.94}."
        )
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of the primary value. E.g. 'count', 'dB SPL', 'm3/m3', 'ppb'."
    )

    # Media
    media_url: Optional[str] = Field(
        default=None,
        description="URL to associated media: audio file, image, spectrogram."
    )
    media_type: Optional[str] = Field(
        default=None,
        description="MIME type of media. E.g. 'audio/mpeg', 'image/jpeg'."
    )

    # Quality
    quality: Quality = Field(default_factory=Quality)

    # Provenance (always preserved)
    provenance: Provenance
    raw: Optional[dict] = Field(
        default=None,
        description="Original API response preserved verbatim for full traceability."
    )

    # Temporal metadata (Phase 1C — supports streaming in Phase 2)
    temporal_resolution: Optional[str] = Field(
        default=None,
        description="Reporting frequency of source. E.g. '15min', 'daily', '5-day', 'event-driven'."
    )

    # Future: intelligence layer
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Cross-modal embedding vector (reserved for Phase 3)."
    )
    related_ids: Optional[list[str]] = Field(
        default=None,
        description="IDs of correlated observations from other sources."
    )


# ---------------------------------------------------------------------------
# Search and adapter support types
# ---------------------------------------------------------------------------

class SearchParams(BaseModel):
    """Parameters for querying observations across adapters."""

    lat: Optional[float] = None
    lng: Optional[float] = Field(default=None, description="Longitude. Alias: lon.")
    lon: Optional[float] = Field(default=None, description="Longitude alias for lng (preferred in scientific contexts).")
    radius_km: Optional[float] = Field(default=None, ge=0)

    def model_post_init(self, __context: Any) -> None:
        """Allow `lon` as an alias for `lng`."""
        if self.lon is not None and self.lng is None:
            self.lng = self.lon
        elif self.lng is not None and self.lon is None:
            self.lon = self.lng
    taxon: Optional[str] = Field(default=None, description="Scientific or common name.")
    taxon_id: Optional[int] = Field(default=None, description="GBIF taxon key.")
    site_id: Optional[str] = Field(default=None, description="Named site code, e.g. 'WREF'.")
    start_date: Optional[str] = Field(default=None, description="ISO 8601 date string.")
    end_date: Optional[str] = Field(default=None, description="ISO 8601 date string.")
    modalities: Optional[list[SignalModality]] = None
    quality_tier_min: Optional[int] = Field(default=None, ge=1, le=4)
    limit: int = Field(default=20, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class SearchRelevance(BaseModel):
    """
    Relevance scoring for search results — the federated ranking layer.

    Every adapter gets consistent ranking for free. Components are exposed
    (not just a scalar) so AI agents can reason about them:
    "geo distance is perfect but taxon match is 0.6 — should I expand to family level?"

    Decided 2026-03-03: retrieve broadly, rank by quality. Never filter at
    retrieval time based on quality — tag, score, return with flags visible.
    """

    score: float = Field(
        ge=0.0, le=1.0,
        description="Composite relevance score (0–1). Weighted combination of components."
    )
    geo_distance_km: Optional[float] = Field(
        default=None,
        description="Distance from query point in kilometres. None if no location query."
    )
    taxon_match: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description=(
            "Taxonomic match score: 1.0=exact species, 0.8=genus, 0.6=family, "
            "0.4=order, 0.0=no match. None if no taxon query."
        )
    )
    temporal_distance_days: Optional[float] = Field(
        default=None,
        description="Distance from query date in days. None if no date query."
    )
    quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Quality component derived from tier + flags. Tier 1=1.0, 2=0.75, 3=0.5, 4=0.25."
    )
    explanation: str = Field(
        description=(
            "Human-readable explanation of the score. "
            "E.g. '3.2km; exact species; tier-2; 2023-06-15'"
        )
    )


class AdapterCapabilities(BaseModel):
    """Self-description of what a data source adapter provides."""

    adapter_id: str
    name: str
    description: str
    modalities: list[SignalModality]
    supports_location_search: bool = False
    supports_taxon_search: bool = False
    supports_date_range: bool = False
    supports_site_search: bool = False
    geographic_coverage: str = "unknown"  # "global", "North America", GeoJSON, etc.
    temporal_coverage_start: Optional[str] = None
    update_frequency: Optional[str] = None  # "real-time", "daily", "static"
    quality_tier: int = Field(ge=1, le=4)
    requires_auth: bool = False
    rate_limit_per_minute: Optional[int] = None
    license: Optional[str] = None
    homepage_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Phase 2: Streaming event types
# ---------------------------------------------------------------------------

AnomalyType = Literal[
    "temperature",     # thermal anomaly
    "flow",            # hydrological flow deviation
    "phenological",    # timing shift (early/late events)
    "acoustic",        # soundscape change
    "spectral",        # vegetation stress (NDVI, EVI)
    "composition",     # species composition shift
    "chemical",        # water/air chemistry deviation
]


class EcologicalAnomaly(BaseModel):
    """A deviation from baseline in an ecological signal.

    Phase 2 type — used by monitoring agents to flag changes.
    """

    id: str = Field(description="Unique anomaly ID, e.g. 'anomaly:russian-river:2026-03-15:flow'")
    anomaly_type: AnomalyType
    location: Location
    detected_at: datetime
    signal_value: float = Field(description="Current observed value")
    baseline_value: float = Field(description="Expected value (30-year normal or seasonal baseline)")
    deviation_pct: float = Field(description="Percentage deviation from baseline")
    severity: Literal["info", "warning", "critical"] = "info"
    description: str = Field(description="Human-readable description of the anomaly")
    sources: list[str] = Field(
        default_factory=list,
        description="Source adapter IDs that contributed to this detection",
    )
    observations: list[str] = Field(
        default_factory=list,
        description="Related EcologicalObservation IDs",
    )
    confidence: float = Field(ge=0, le=1, description="Detection confidence 0-1")


EcologicalEventType = Literal[
    "drought_cascade",       # multi-signal drought pattern
    "phenological_shift",    # timing mismatch between species/seasons
    "range_expansion",       # species appearing in new areas
    "die_off",               # mass mortality event
    "bloom",                 # algal bloom, insect emergence
    "migration",             # seasonal movement pattern
    "tipping_point_warning", # approaching irreversible change
]


class EcologicalEvent(BaseModel):
    """A classified ecological event synthesized from multiple anomalies.

    Phase 2-3 type — higher-level interpretation of raw signals.
    """

    id: str = Field(description="Unique event ID")
    event_type: EcologicalEventType
    location: Location
    detected_at: datetime
    duration_days: Optional[int] = None
    severity: Literal["info", "warning", "critical", "emergency"] = "info"
    title: str = Field(description="Short event title, e.g. 'Drought cascade in Klamath Basin'")
    narrative: str = Field(description="Multi-sentence, data-grounded description of the event")
    anomalies: list[str] = Field(
        default_factory=list,
        description="Related EcologicalAnomaly IDs",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="All source adapter IDs that contributed",
    )
    confidence: float = Field(ge=0, le=1)
    historical_analog: Optional[str] = Field(
        default=None,
        description="Reference to a similar past event, e.g. '2021 Klamath drought cascade'",
    )


# ---------------------------------------------------------------------------
# Phase 3-5: Ecosystem state (continuous monitoring)
# ---------------------------------------------------------------------------

class EcosystemState(BaseModel):
    """Real-time ecological state vector for a monitored location.

    Phase 3+ type — the foundation for emergent intelligence. When
    maintained continuously for many ecosystems, cross-location pattern
    detection becomes possible.
    """

    id: str = Field(description="Ecosystem identifier, e.g. 'watershed:russian-river'")
    location: Location
    timestamp: datetime
    period_days: int = Field(default=30, description="Lookback window for computed metrics")

    # Hydrological signals
    streamflow_cfs: Optional[float] = Field(default=None, description="Current streamflow (cubic feet/sec)")
    streamflow_baseline: Optional[float] = Field(default=None, description="30-year normal for this date")
    soil_moisture_pct: Optional[float] = Field(default=None, description="Volumetric water content")
    precipitation_7d_mm: Optional[float] = Field(default=None, description="7-day precipitation total")

    # Biological signals
    species_richness: Optional[int] = Field(default=None, description="Species count in lookback window")
    species_baseline: Optional[int] = Field(default=None, description="Expected species count for this date")
    acoustic_diversity_index: Optional[float] = Field(default=None, description="Acoustic complexity index")

    # Spectral signals
    ndvi: Optional[float] = Field(default=None, description="Normalized Difference Vegetation Index")
    ndvi_baseline: Optional[float] = Field(default=None, description="Seasonal NDVI baseline")

    # Climate signals
    temp_mean_c: Optional[float] = Field(default=None, description="Mean temperature over period")
    temp_anomaly_c: Optional[float] = Field(default=None, description="Deviation from 30-year normal")

    # Derived metrics
    deviation_vector: list[float] = Field(
        default_factory=list,
        description="Normalized deviation from baseline for each signal (-1 to 1)",
    )
    overall_health_score: Optional[float] = Field(
        default=None,
        ge=0, le=100,
        description="Composite ecosystem health score (0-100)",
    )
    trend_direction: Optional[Literal["improving", "stable", "declining", "critical"]] = None
    active_anomalies: list[str] = Field(
        default_factory=list,
        description="IDs of currently active EcologicalAnomaly objects",
    )
    active_events: list[str] = Field(
        default_factory=list,
        description="IDs of currently active EcologicalEvent objects",
    )

    # Data provenance
    sources_contributing: list[str] = Field(
        default_factory=list,
        description="Adapter IDs currently feeding this state vector",
    )
    last_updated: Optional[datetime] = None
