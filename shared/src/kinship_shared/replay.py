"""
History replay and scoring for the flow detector (spec 016).

Replays a gauge's daily record day by day, judging each day only against
years before it (no look-ahead), groups flagged days into episodes, and
scores those episodes against a cited list of documented events.

Unmatched episodes are NOT false alarms: the reference list is short, so
many of them are real events nobody has listed yet.
"""

from __future__ import annotations

import bisect
from datetime import date, timedelta
from statistics import median
from typing import Literal, Optional

from pydantic import BaseModel, Field

Flag = Literal["low", "high"]

_DOYS = range(1, 367)


class ReplayDay(BaseModel):
    day: date
    flow_cfs: float
    percentile: Optional[float] = None
    flag: Optional[Flag] = None


class Episode(BaseModel):
    kind: Flag
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


class ReferenceEvent(BaseModel):
    id: str
    kind: Literal["low_flow", "high_flow"]
    site_id: str
    start: date
    end: date
    title: str = ""
    source: str = ""
    note: str = ""

    @property
    def flag(self) -> Flag:
        return "low" if self.kind == "low_flow" else "high"


class Scorecard(BaseModel):
    site_id: str
    low_pct: float
    high_pct: float
    years_replayed: int
    events_total: int = Field(description="Events with replayed data in their window")
    events_hit: int
    events_no_data: list[str] = Field(
        default_factory=list,
        description="Events whose dates have no replayed values (gauge gap); not counted as misses",
    )
    lead_days_median: Optional[float] = Field(
        default=None,
        description="First flag relative to documented start; negative = flagged before",
    )
    episodes_total: int
    episodes_unmatched: int
    hits: dict[str, Optional[int]] = Field(
        default_factory=dict, description="event id -> lead days (None = missed)"
    )
    unmatched: list[Episode] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _doy_distance(a: int, b: int) -> int:
    d = abs(a - b)
    return min(d, 366 - d)


def _neighbours(window_days: int) -> dict[int, list[int]]:
    """For each day-of-year, the days-of-year whose ±window includes it."""
    return {k: [t for t in _DOYS if _doy_distance(k, t) <= window_days] for k in _DOYS}


def replay_flow(
    record: list[tuple[date, float]],
    start: date,
    end: date,
    window_days: int = 7,
    min_years: int = 10,
    low_pct: float = 10,
    high_pct: float = 90,
) -> list[ReplayDay]:
    """Day-by-day percentile of flow against prior years only.

    A day in year Y is compared with values within ±window_days of its
    day-of-year from years strictly before Y, the same rule as
    ``baselines.flow_percentile`` but with the future removed. Days with
    fewer than ``min_years`` prior years in the window get percentile None
    and no flag.
    """
    record = sorted((d, v) for d, v in record if v is not None)
    neighbours = _neighbours(window_days)
    pool: dict[int, list[float]] = {t: [] for t in _DOYS}  # sorted values
    pool_years: dict[int, set[int]] = {t: set() for t in _DOYS}

    out: list[ReplayDay] = []
    i = 0
    while i < len(record):
        year = record[i][0].year
        j = i
        while j < len(record) and record[j][0].year == year:
            j += 1
        this_year = record[i:j]

        for d, v in this_year:
            if not (start <= d <= end):
                continue
            t = d.timetuple().tm_yday
            ref = pool[t]
            pct: Optional[float] = None
            flag: Optional[Flag] = None
            if ref and len(pool_years[t]) >= min_years:
                below = bisect.bisect_left(ref, v)
                equal = bisect.bisect_right(ref, v) - below
                pct = round(100.0 * (below + 0.5 * equal) / len(ref), 1)
                if pct <= low_pct:
                    flag = "low"
                elif pct >= high_pct:
                    flag = "high"
            out.append(ReplayDay(day=d, flow_cfs=v, percentile=pct, flag=flag))

        # Only now does this year become "the past".
        for d, v in this_year:
            for t in neighbours[d.timetuple().tm_yday]:
                bisect.insort(pool[t], v)
                pool_years[t].add(year)
        i = j
    return out


def group_episodes(
    days: list[ReplayDay], min_low_days: int = 14, min_high_days: int = 2
) -> list[Episode]:
    """Runs of consecutive calendar days with the same flag, kept if long enough."""
    episodes: list[Episode] = []
    run_kind: Optional[Flag] = None
    run_start = run_end = None

    def close():
        if run_kind is None:
            return
        need = min_low_days if run_kind == "low" else min_high_days
        if (run_end - run_start).days + 1 >= need:
            episodes.append(Episode(kind=run_kind, start=run_start, end=run_end))

    for rd in days:
        contiguous = run_end is not None and rd.day == run_end + timedelta(days=1)
        if rd.flag is not None and rd.flag == run_kind and contiguous:
            run_end = rd.day
            continue
        close()
        run_kind = rd.flag
        run_start = run_end = rd.day if rd.flag else None
        if rd.flag is None:
            run_kind = None
    close()
    return episodes


def _overlaps(a_start: date, a_end: date, b_start: date, b_end: date) -> bool:
    return a_start <= b_end and b_start <= a_end


def score(
    site_id: str,
    days: list[ReplayDay],
    episodes: list[Episode],
    events: list[ReferenceEvent],
    low_pct: float = 10,
    high_pct: float = 90,
) -> Scorecard:
    """Score episodes against documented events at one gauge.

    An event is hit when an episode of the matching kind overlaps it; lead
    is the first such episode's start minus the event's start. An episode
    is unmatched when it overlaps no event of its kind.
    """
    events = [e for e in events if e.site_id == site_id]
    judged = {rd.day for rd in days if rd.percentile is not None}
    no_data = [e.id for e in events
               if not any(e.start + timedelta(days=k) in judged
                          for k in range((e.end - e.start).days + 1))]
    events = [e for e in events if e.id not in no_data]
    hits: dict[str, Optional[int]] = {}
    leads: list[int] = []
    for ev in events:
        matching = [ep for ep in episodes
                    if ep.kind == ev.flag and _overlaps(ep.start, ep.end, ev.start, ev.end)]
        if matching:
            lead = (min(ep.start for ep in matching) - ev.start).days
            hits[ev.id] = lead
            leads.append(lead)
        else:
            hits[ev.id] = None

    unmatched = [ep for ep in episodes
                 if not any(ep.kind == ev.flag and _overlaps(ep.start, ep.end, ev.start, ev.end)
                            for ev in events)]
    years = {rd.day.year for rd in days if rd.percentile is not None}
    notes = []
    if no_data:
        notes.append(
            "No daily data on the documented dates (gauge gap), so not scored: "
            + ", ".join(no_data)
        )
    if unmatched:
        notes.append(
            "Unmatched episodes are not false alarms: the reference list is short, "
            "so many are likely real events that nobody has listed yet."
        )
    return Scorecard(
        site_id=site_id, low_pct=low_pct, high_pct=high_pct,
        years_replayed=len(years),
        events_total=len(events), events_no_data=no_data, events_hit=sum(1 for v in hits.values() if v is not None),
        lead_days_median=float(median(leads)) if leads else None,
        episodes_total=len(episodes), episodes_unmatched=len(unmatched),
        hits=hits, unmatched=unmatched, notes=notes,
    )
