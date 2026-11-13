"""
Anomaly detection pipeline for ecological signals.

Each detector takes current data and baselines, and emits zero or more
EcologicalAnomaly objects. Detectors are composable — the pipeline runs
all relevant detectors for a location and aggregates results.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from .baselines import BaselineValues, compute_deviation
from .schema import AnomalyType, EcologicalAnomaly, EcologicalObservation, EcosystemState, Location

logger = logging.getLogger(__name__)


class AnomalyDetector(ABC):
    @property
    @abstractmethod
    def anomaly_type(self) -> AnomalyType: ...

    @abstractmethod
    def detect(
        self, *, location: Location,
        observations: list[EcologicalObservation],
        baseline: BaselineValues | None = None,
        state: EcosystemState | None = None,
    ) -> list[EcologicalAnomaly]: ...


def _make_anomaly_id(anomaly_type: str, lat: float, lng: float, dt: datetime) -> str:
    return f"anomaly:{lat:.2f}_{lng:.2f}:{dt.strftime('%Y-%m-%d')}:{anomaly_type}"


def _classify_severity(deviation_pct: float) -> str:
    abs_dev = abs(deviation_pct)
    if abs_dev > 100:
        return "critical"
    elif abs_dev > 50:
        return "warning"
    return "info"


class TemperatureDetector(AnomalyDetector):
    @property
    def anomaly_type(self) -> AnomalyType:
        return "temperature"

    def detect(self, *, location, observations, baseline=None, state=None):
        if state is None or state.temp_mean_c is None:
            return []
        if baseline is None or baseline.temp_mean_c is None:
            return []

        std = baseline.temp_std_c or 2.0
        z = compute_deviation(state.temp_mean_c, baseline.temp_mean_c, std)

        if abs(z) < 2.0:
            return []

        deviation_pct = ((state.temp_mean_c - baseline.temp_mean_c) / max(abs(baseline.temp_mean_c), 0.1)) * 100
        direction = "above" if z > 0 else "below"

        return [EcologicalAnomaly(
            id=_make_anomaly_id("temperature", location.lat, location.lng, state.timestamp),
            anomaly_type="temperature",
            location=location,
            detected_at=state.timestamp,
            signal_value=state.temp_mean_c,
            baseline_value=baseline.temp_mean_c,
            deviation_pct=round(deviation_pct, 1),
            severity=_classify_severity(deviation_pct),
            description=f"Temperature {abs(z):.1f} sigma {direction} normal: {state.temp_mean_c:.1f}°C vs baseline {baseline.temp_mean_c:.1f}°C",
            sources=state.sources_contributing,
            confidence=min(1.0, abs(z) / 4.0),
        )]


class FlowDetector(AnomalyDetector):
    @property
    def anomaly_type(self) -> AnomalyType:
        return "flow"

    def detect(self, *, location, observations, baseline=None, state=None):
        if state is not None and state.streamflow_percentile is not None:
            return self._detect_by_percentile(location, state)
        if state is None or state.streamflow_cfs is None or state.streamflow_baseline is None:
            return []
        if state.streamflow_baseline < 1.0:
            return []

        deviation_pct = ((state.streamflow_cfs - state.streamflow_baseline) / state.streamflow_baseline) * 100

        if abs(deviation_pct) < 50:
            return []

        direction = "above" if deviation_pct > 0 else "below"
        signal = "flood risk" if deviation_pct > 0 else "drought signal"

        return [EcologicalAnomaly(
            id=_make_anomaly_id("flow", location.lat, location.lng, state.timestamp),
            anomaly_type="flow",
            location=location,
            detected_at=state.timestamp,
            signal_value=state.streamflow_cfs,
            baseline_value=state.streamflow_baseline,
            deviation_pct=round(deviation_pct, 1),
            severity=_classify_severity(deviation_pct),
            description=f"Streamflow {abs(deviation_pct):.0f}% {direction} normal ({signal}): {state.streamflow_cfs:.0f} cfs vs baseline {state.streamflow_baseline:.0f} cfs",
            sources=["usgs-nwis"],
            confidence=min(1.0, abs(deviation_pct) / 200.0),
        )]


    @staticmethod
    def _ordinal(n: float) -> str:
        i = int(round(n))
        suffix = "th" if 10 <= i % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
        return f"{i}{suffix}"

    def _detect_by_percentile(self, location, state):
        """Flow against the gauge's own day-of-year record (spec 015)."""
        pct = state.streamflow_percentile
        if 10 < pct < 90:
            return []
        if pct <= 2 or pct >= 98:
            severity = "critical"
        elif pct <= 5 or pct >= 95:
            severity = "warning"
        else:
            severity = "info"

        median = state.streamflow_baseline
        deviation_pct = None
        if state.streamflow_cfs is not None and median:
            deviation_pct = round((state.streamflow_cfs - median) / median * 100, 1)

        record = []
        if state.streamflow_record_years:
            record.append(f"{state.streamflow_record_years} years of record")
        if state.streamflow_site_id:
            record.append(state.streamflow_site_id)
        description = f"Streamflow at the {self._ordinal(pct)} percentile for this date"
        if record:
            description += f" ({', '.join(record)})"
        if state.streamflow_cfs is not None and median is not None:
            description += f": {state.streamflow_cfs:.0f} cfs vs median {median:.0f} cfs"

        return [EcologicalAnomaly(
            id=_make_anomaly_id("flow", location.lat, location.lng, state.timestamp),
            anomaly_type="flow",
            location=location,
            detected_at=state.timestamp,
            signal_value=state.streamflow_cfs if state.streamflow_cfs is not None else pct,
            baseline_value=median if median is not None else 50.0,
            deviation_pct=deviation_pct if deviation_pct is not None else round(pct - 50.0, 1),
            severity=severity,
            description=description,
            sources=["usgs-nwis"],
            confidence=round(min(1.0, abs(pct - 50.0) / 50.0), 2),
        )]


class CompositionDetector(AnomalyDetector):
    @property
    def anomaly_type(self) -> AnomalyType:
        return "composition"

    def detect(self, *, location, observations, baseline=None, state=None):
        if state is None or state.species_richness is None or state.species_baseline is None:
            return []
        if state.species_baseline < 1:
            return []

        deviation_pct = ((state.species_richness - state.species_baseline) / state.species_baseline) * 100

        if abs(deviation_pct) < 40:
            return []

        direction = "increase" if deviation_pct > 0 else "decrease"

        return [EcologicalAnomaly(
            id=_make_anomaly_id("composition", location.lat, location.lng, state.timestamp),
            anomaly_type="composition",
            location=location,
            detected_at=state.timestamp,
            signal_value=float(state.species_richness),
            baseline_value=float(state.species_baseline),
            deviation_pct=round(deviation_pct, 1),
            severity=_classify_severity(deviation_pct),
            description=f"Species richness {direction}: {state.species_richness} species vs baseline {state.species_baseline} ({abs(deviation_pct):.0f}% change)",
            sources=state.sources_contributing,
            confidence=min(1.0, abs(deviation_pct) / 150.0),
        )]


class PhenologicalDetector(AnomalyDetector):
    @property
    def anomaly_type(self) -> AnomalyType:
        return "phenological"

    def detect(self, *, location, observations, baseline=None, state=None):
        if not observations or len(observations) < 3:
            return []
        if not baseline or not baseline.species_richness_mean:
            return []
        if not state or not state.species_richness:
            return []

        now = datetime.now(timezone.utc)
        early_late_ratio = state.species_richness / max(baseline.species_richness_mean, 1)
        if 0.5 <= early_late_ratio <= 1.5:
            return []

        deviation_pct = (early_late_ratio - 1.0) * 100
        return [EcologicalAnomaly(
            id=_make_anomaly_id("phenological", location.lat, location.lng, now),
            anomaly_type="phenological",
            location=location,
            detected_at=now,
            signal_value=float(state.species_richness),
            baseline_value=float(baseline.species_richness_mean),
            deviation_pct=round(deviation_pct, 1),
            severity=_classify_severity(deviation_pct),
            description=f"Unusual species activity for this time of year: {state.species_richness} species vs expected {baseline.species_richness_mean}",
            sources=state.sources_contributing if state else [],
            confidence=0.5,
        )]


DEFAULT_DETECTORS: list[AnomalyDetector] = [
    TemperatureDetector(),
    FlowDetector(),
    CompositionDetector(),
    PhenologicalDetector(),
]


def run_anomaly_detection(
    *, location: Location,
    observations: list[EcologicalObservation] | None = None,
    baseline: BaselineValues | None = None,
    state: EcosystemState | None = None,
    detectors: list[AnomalyDetector] | None = None,
) -> list[EcologicalAnomaly]:
    """Run all anomaly detectors and return aggregated results."""
    detectors = detectors or DEFAULT_DETECTORS
    all_anomalies: list[EcologicalAnomaly] = []

    for detector in detectors:
        try:
            found = detector.detect(
                location=location, observations=observations or [],
                baseline=baseline, state=state,
            )
            all_anomalies.extend(found)
        except Exception as e:
            logger.warning("Detector %s failed: %s", detector.anomaly_type, e)

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    all_anomalies.sort(key=lambda a: (severity_order.get(a.severity, 3), -a.confidence))
    return all_anomalies
