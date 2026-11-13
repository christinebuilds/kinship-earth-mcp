"""
Builds an EcosystemState for a location by querying live data sources
and comparing against baselines.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from .baselines import (
    compute_baselines_from_era5,
    compute_baselines_from_usgs,
    compute_deviation,
    compute_health_score,
    classify_trend,
)
from .schema import EcosystemState, Location, SearchParams

logger = logging.getLogger(__name__)


async def build_ecosystem_state(
    *,
    site_id: str,
    location: Location,
    era5_adapter,
    usgs_adapter,
    ebird_adapter=None,
    gbif_adapter=None,
    period_days: int = 30,
    baseline_years: int = 30,
    recent_health_scores: list[float] | None = None,
) -> EcosystemState:
    """Build a current EcosystemState for a location."""
    now = datetime.now(timezone.utc)
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=period_days)).strftime("%Y-%m-%d")

    temp_mean = None
    precip_7d = None
    streamflow = None
    species_richness = None
    sources = []

    # 1. ERA5 climate
    try:
        era5_data = await era5_adapter.get_daily(lat=location.lat, lng=location.lng, start_date=start_date, end_date=end_date)
        daily = era5_data.get("daily", {})
        temps = [v for v in daily.get("temperature_2m_mean", []) if v is not None]
        if temps:
            temp_mean = round(sum(temps) / len(temps), 2)
        precips = [v for v in daily.get("precipitation_sum", []) if v is not None]
        if precips:
            precip_7d = round(sum(precips[-7:]), 2)
        sources.append("era5")
    except Exception as e:
        logger.warning("ERA5 query failed for %s: %s", site_id, e)

    # 2. USGS hydrology
    try:
        usgs_obs = await usgs_adapter.search(
            SearchParams(lat=location.lat, lng=location.lng, radius_km=50.0,
                         start_date=(now - timedelta(days=7)).strftime("%Y-%m-%d"),
                         end_date=end_date, limit=20)
        )
        flows = []
        for obs in usgs_obs:
            if obs.value and isinstance(obs.value, dict):
                flow = obs.value.get("discharge_cfs") or obs.value.get("value")
                if flow is not None:
                    try:
                        flows.append(float(flow))
                    except (ValueError, TypeError):
                        continue
        if flows:
            streamflow = round(sum(flows) / len(flows), 2)
            sources.append("usgs-nwis")
    except Exception as e:
        logger.warning("USGS query failed for %s: %s", site_id, e)

    # 3. Species richness
    bio_adapter = ebird_adapter or gbif_adapter
    if bio_adapter:
        try:
            bio_obs = await bio_adapter.search(
                SearchParams(lat=location.lat, lng=location.lng, radius_km=25.0,
                             start_date=(now - timedelta(days=period_days)).strftime("%Y-%m-%d"),
                             end_date=end_date, limit=200)
            )
            species_names = {obs.taxon.scientific_name for obs in bio_obs if obs.taxon and obs.taxon.scientific_name}
            if species_names:
                species_richness = len(species_names)
                sources.append("ebird" if ebird_adapter else "gbif")
        except Exception as e:
            logger.warning("Biology query failed for %s: %s", site_id, e)

    # 4. Baselines
    baseline = await compute_baselines_from_era5(era5_adapter, location.lat, location.lng, now, years_back=baseline_years)
    usgs_baseline = await compute_baselines_from_usgs(
        usgs_adapter, location.lat, location.lng, now, current_flow=streamflow
    )

    # 5. Deviations
    deviations = []
    temp_anomaly = None

    if temp_mean is not None and baseline.temp_mean_c is not None:
        temp_anomaly = round(temp_mean - baseline.temp_mean_c, 2)
        d = compute_deviation(temp_mean, baseline.temp_mean_c, baseline.temp_std_c or 2.0)
        deviations.append(round(max(-1.0, min(1.0, d / 3.0)), 3))

    streamflow_baseline = usgs_baseline.get("streamflow_mean_cfs")
    streamflow_percentile = usgs_baseline.get("streamflow_percentile")
    streamflow_baseline_kind = usgs_baseline.get("baseline_kind")

    if streamflow is not None and streamflow_baseline is not None:
        std = usgs_baseline.get("streamflow_std_cfs", streamflow_baseline * 0.3)
        d = compute_deviation(streamflow, streamflow_baseline, std)
        deviations.append(round(max(-1.0, min(1.0, d / 3.0)), 3))

    # 6. Health score and trend
    health = compute_health_score(deviations)
    health_history = list(recent_health_scores or [])
    health_history.append(health)
    trend = classify_trend(health_history)

    return EcosystemState(
        id=site_id,
        location=location,
        timestamp=now,
        period_days=period_days,
        streamflow_cfs=streamflow,
        streamflow_baseline=streamflow_baseline,
        streamflow_percentile=streamflow_percentile,
        streamflow_baseline_kind=streamflow_baseline_kind,
        precipitation_7d_mm=precip_7d,
        species_richness=species_richness,
        species_baseline=baseline.species_richness_mean,
        temp_mean_c=temp_mean,
        temp_anomaly_c=temp_anomaly,
        deviation_vector=deviations,
        overall_health_score=health,
        trend_direction=trend,
        sources_contributing=sources,
        last_updated=now,
    )
