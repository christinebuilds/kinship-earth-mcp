"""
"What changed near home?" — compare what was reported in one place across
two stretches of days, and against the same stretch last year (spec 022).

Pure functions over EcologicalObservation lists. Counts are sightings
reported by volunteers, not animals: a change can reflect who went out
looking as much as what was there, and the wording here says so.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .schema import EcologicalObservation

_GROUPS = {
    "aves": "bird", "plantae": "plant", "insecta": "insect", "mammalia": "mammal",
    "reptilia": "reptile", "amphibia": "amphibian", "fungi": "fungus",
    "arachnida": "arachnid", "actinopterygii": "fish", "mollusca": "mollusc",
}


class SpeciesTally(BaseModel):
    scientific_name: str
    common_name: Optional[str] = None
    taxon_group: Optional[str] = None
    n_observations: int
    first_seen: date
    last_seen: date
    sources: list[str] = Field(default_factory=list)


class SpeciesWindow(BaseModel):
    """What was seen in one place over one stretch of days."""

    start: date
    end: date
    species: dict[str, SpeciesTally] = Field(default_factory=dict)

    @property
    def n_observations(self) -> int:
        return sum(t.n_observations for t in self.species.values())


class SeasonalChange(BaseModel):
    scientific_name: str
    common_name: Optional[str] = None
    taxon_group: Optional[str] = None
    kind: Literal["arrived", "departed", "more_often", "less_often"]
    n_now: int
    n_before: int
    first_seen: Optional[date] = None
    last_seen: Optional[date] = None
    last_year: Literal["also_present", "absent", "unknown"] = "unknown"
    timing_vs_last_year_days: Optional[int] = None
    note: str


def _taxon_group(obs: EcologicalObservation) -> Optional[str]:
    source = obs.provenance.source_api if obs.provenance else None
    if source == "ebird":
        return "bird"
    iconic = obs.value.get("iconic_taxon") if isinstance(obs.value, dict) else None
    for key in (iconic, getattr(obs.taxon, "class_", None), getattr(obs.taxon, "kingdom", None)):
        if key and key.lower() in _GROUPS:
            return _GROUPS[key.lower()]
    return None


def tally(observations: list[EcologicalObservation], start: date, end: date) -> SpeciesWindow:
    """Count observations per species within [start, end], merging sources."""
    window = SpeciesWindow(start=start, end=end)
    for obs in observations:
        if not obs.taxon or not obs.taxon.scientific_name or obs.observed_at is None:
            continue
        day = obs.observed_at.date()
        if day < start or day > end:
            continue
        name = obs.taxon.scientific_name
        source = obs.provenance.source_api if obs.provenance else "unknown"
        entry = window.species.get(name)
        if entry is None:
            window.species[name] = SpeciesTally(
                scientific_name=name,
                common_name=obs.taxon.common_name,
                taxon_group=_taxon_group(obs),
                n_observations=1,
                first_seen=day,
                last_seen=day,
                sources=[source],
            )
            continue
        entry.n_observations += 1
        entry.first_seen = min(entry.first_seen, day)
        entry.last_seen = max(entry.last_seen, day)
        entry.common_name = entry.common_name or obs.taxon.common_name
        entry.taxon_group = entry.taxon_group or _taxon_group(obs)
        if source not in entry.sources:
            entry.sources.append(source)
    return window


def _label(t: SpeciesTally) -> str:
    return t.common_name or t.scientific_name


def _last_year_status(name: str, last_year: Optional[SpeciesWindow]) -> str:
    if last_year is None:
        return "unknown"
    return "also_present" if name in last_year.species else "absent"


def _timing_note(days: int) -> str:
    if days == 0:
        return "the same time of year as last year"
    word = "earlier" if days < 0 else "later"
    n = abs(days)
    return f"{n} day{'' if n == 1 else 's'} {word} than last year"


def seasonal_changes(
    now: SpeciesWindow,
    before: SpeciesWindow,
    last_year: Optional[SpeciesWindow] = None,
    min_observations: int = 2,
) -> list[SeasonalChange]:
    """Arrivals, departures and large shifts in how often species are reported."""
    changes: list[SeasonalChange] = []

    for name, t in now.species.items():
        status = _last_year_status(name, last_year)
        prev = before.species.get(name)
        if prev is None:
            if t.n_observations < min_observations:
                continue
            timing = None
            note = f"{_label(t)} was reported {t.n_observations} times, first on {t.first_seen.isoformat()}, and not in the period before."
            if status == "also_present":
                ly = last_year.species[name]
                timing = t.first_seen.timetuple().tm_yday - ly.first_seen.timetuple().tm_yday
                note += f" First report is {_timing_note(timing)}."
            elif status == "absent":
                note += " It was not reported here at this time last year."
            changes.append(SeasonalChange(
                scientific_name=name, common_name=t.common_name, taxon_group=t.taxon_group,
                kind="arrived", n_now=t.n_observations, n_before=0,
                first_seen=t.first_seen, last_year=status,
                timing_vs_last_year_days=timing, note=note,
            ))
            continue

        hi, lo = max(t.n_observations, prev.n_observations), min(t.n_observations, prev.n_observations)
        if hi >= 3 * lo and hi - lo >= 5:
            more = t.n_observations > prev.n_observations
            changes.append(SeasonalChange(
                scientific_name=name, common_name=t.common_name, taxon_group=t.taxon_group,
                kind="more_often" if more else "less_often",
                n_now=t.n_observations, n_before=prev.n_observations, last_year=status,
                note=(f"{_label(t)} is being reported {'more' if more else 'less'} often: "
                      f"{t.n_observations} reports now, {prev.n_observations} in the period before."),
            ))

    for name, prev in before.species.items():
        if name in now.species or prev.n_observations < min_observations:
            continue
        changes.append(SeasonalChange(
            scientific_name=name, common_name=prev.common_name, taxon_group=prev.taxon_group,
            kind="departed", n_now=0, n_before=prev.n_observations,
            last_seen=prev.last_seen, last_year=_last_year_status(name, last_year),
            note=(f"{_label(prev)} was reported {prev.n_observations} times in the period before, "
                  f"last on {prev.last_seen.isoformat()}, and not since."),
        ))

    order = {"arrived": 0, "departed": 1, "more_often": 2, "less_often": 3}
    changes.sort(key=lambda c: (order[c.kind], -max(c.n_now, c.n_before), c.scientific_name))
    return changes
