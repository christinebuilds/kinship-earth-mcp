"""Spec 016: no-look-ahead replay, episode grouping and scoring (offline, synthetic)."""

import math
import random
from datetime import date, timedelta

from kinship_shared.replay import (
    Episode, ReferenceEvent, ReplayDay, group_episodes, replay_flow, score,
)


def _record(years=50, start_year=1970, drought=None, seed=1):
    """Seasonal flow with noise; ``drought`` = (start, days) cut to 5% of normal."""
    rng = random.Random(seed)
    out, d = [], date(start_year, 1, 1)
    end = date(start_year + years - 1, 12, 31)
    while d <= end:
        doy = d.timetuple().tm_yday
        v = (500 + 400 * math.cos(2 * math.pi * (doy - 15) / 365)) * rng.uniform(0.7, 1.3)
        if drought and drought[0] <= d < drought[0] + timedelta(days=drought[1]):
            v *= 0.05
        out.append((d, round(v, 2)))
        d += timedelta(days=1)
    return out


def test_injected_drought_is_flagged_and_scored_as_hit():
    dstart = date(2009, 7, 1)  # year 40 of 1970-2019
    rec = _record(drought=(dstart, 60))
    days = replay_flow(rec, date(1970, 1, 1), date(2019, 12, 31))
    eps = group_episodes(days)
    low = [e for e in eps if e.kind == "low" and e.start <= dstart + timedelta(days=59) and e.end >= dstart]
    assert low and low[0].days >= 55
    ev = ReferenceEvent(id="synthetic-2009", kind="low_flow", site_id="S",
                        start=dstart, end=dstart + timedelta(days=59))
    card = score("S", days, eps, [ev])
    assert (card.events_hit, card.events_total) == (1, 1)
    assert card.hits["synthetic-2009"] is not None and abs(card.hits["synthetic-2009"]) <= 3


def test_no_look_ahead():
    rec = _record(years=30)
    probe = date(1995, 6, 15)
    before = {r.day: r.percentile for r in replay_flow(rec, probe, probe)}
    future_changed = [(d, v * 100 if d > probe else v) for d, v in rec]
    after = {r.day: r.percentile for r in replay_flow(future_changed, probe, probe)}
    assert before[probe] is not None and before == after


def test_first_years_have_no_percentile():
    rec = _record(years=15)
    days = replay_flow(rec, date(1970, 1, 1), date(1984, 12, 31), min_years=10)
    early = [r for r in days if r.day.year < 1980]
    assert early and all(r.percentile is None and r.flag is None for r in early)
    assert any(r.percentile is not None for r in days if r.day.year >= 1981)


def _days(flags, start=date(2000, 1, 1)):
    return [ReplayDay(day=start + timedelta(days=i), flow_cfs=1.0, percentile=50.0, flag=f)
            for i, f in enumerate(flags)]


def test_episode_minimum_lengths():
    flags = ["low"] * 13 + [None] + ["low"] * 14 + [None] + ["high"] * 1 + [None] + ["high"] * 2
    eps = group_episodes(_days(flags), min_low_days=14, min_high_days=2)
    assert [(e.kind, e.days) for e in eps] == [("low", 14), ("high", 2)]
    eps = group_episodes(_days(flags), min_low_days=13, min_high_days=1)
    assert [(e.kind, e.days) for e in eps] == [("low", 13), ("low", 14), ("high", 1), ("high", 2)]


def test_unmatched_episode_does_not_change_event_count():
    days = _days(["low"] * 20)
    lone = Episode(kind="high", start=date(2001, 1, 1), end=date(2001, 1, 5))
    matched = Episode(kind="low", start=date(2000, 1, 1), end=date(2000, 1, 20))
    ev = ReferenceEvent(id="e", kind="low_flow", site_id="S",
                        start=date(2000, 1, 10), end=date(2000, 2, 1))
    card = score("S", days, [matched, lone], [ev])
    assert card.events_total == 1 and card.events_hit == 1
    assert card.hits["e"] == -9
    assert card.episodes_unmatched == 1 and card.unmatched[0].kind == "high"
    assert any("not false alarms" in n for n in card.notes)
