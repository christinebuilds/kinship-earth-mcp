"""Tests for "what changed near home" (spec 022). Offline, synthetic observations."""

from datetime import date, datetime, timezone

from kinship_shared.schema import EcologicalObservation, Location, Provenance, Quality, TaxonInfo
from kinship_shared.seasonal_changes import seasonal_changes, tally

_n = 0


def obs(name: str, day: date, source: str = "inaturalist", common: str | None = None):
    global _n
    _n += 1
    return EcologicalObservation(
        id=f"{source}:{_n}",
        modality="occurrence",
        location=Location(lat=37.76, lng=-122.42),
        observed_at=datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc),
        taxon=TaxonInfo(scientific_name=name, common_name=common),
        quality=Quality(tier=2),
        provenance=Provenance(source_api=source, source_id=str(_n), license="CC-BY-NC"),
    )


NOW = (date(2026, 3, 8), date(2026, 3, 21))
BEFORE = (date(2026, 2, 22), date(2026, 3, 7))
LAST_YEAR = (date(2025, 3, 1), date(2025, 3, 28))


def windows(now_obs, before_obs, ly_obs=None):
    return (
        tally(now_obs, *NOW),
        tally(before_obs, *BEFORE),
        tally(ly_obs, *LAST_YEAR) if ly_obs is not None else None,
    )


def test_arrived():
    now, before, _ = windows(
        [obs("Selasphorus sasin", date(2026, 3, d), common="Allen's Hummingbird") for d in (10, 12, 15)], [])
    changes = seasonal_changes(now, before)
    assert len(changes) == 1
    c = changes[0]
    assert c.kind == "arrived" and c.n_now == 3 and c.n_before == 0
    assert c.first_seen == date(2026, 3, 10)
    assert "Allen's Hummingbird" in c.note


def test_departed():
    now, before, _ = windows([], [obs("Zonotrichia atricapilla", date(2026, 3, d)) for d in (1, 3, 5)])
    changes = seasonal_changes(now, before)
    assert [c.kind for c in changes] == ["departed"]
    assert changes[0].last_seen == date(2026, 3, 5)


def test_single_sighting_is_not_a_change():
    now, before, _ = windows([obs("Rara avis", date(2026, 3, 10))], [obs("Olim visa", date(2026, 3, 1))])
    assert seasonal_changes(now, before) == []


def test_more_often_needs_a_large_shift():
    def n(name, count, day):
        return [obs(name, day) for _ in range(count)]

    now, before, _ = windows(
        n("Calypte anna", 12, date(2026, 3, 10)) + n("Sayornis nigricans", 4, date(2026, 3, 10)),
        n("Calypte anna", 2, date(2026, 3, 1)) + n("Sayornis nigricans", 2, date(2026, 3, 1)),
    )
    changes = seasonal_changes(now, before)
    assert [(c.scientific_name, c.kind) for c in changes] == [("Calypte anna", "more_often")]
    assert "reported" in changes[0].note and "abundant" not in changes[0].note


def test_timing_against_last_year():
    now, before, ly = windows(
        [obs("Hirundo rustica", date(2026, 3, d)) for d in (10, 11)],
        [],
        [obs("Hirundo rustica", date(2025, 3, 20))],
    )
    c = seasonal_changes(now, before, ly)[0]
    assert c.last_year == "also_present"
    assert c.timing_vs_last_year_days == -10
    assert "10 days earlier than last year" in c.note


def test_absent_last_year_and_unknown():
    now, before, ly = windows([obs("Nova species", date(2026, 3, d)) for d in (10, 11)], [], [])
    assert seasonal_changes(now, before, ly)[0].last_year == "absent"
    assert all(c.last_year == "unknown" for c in seasonal_changes(now, before, None))


def test_tally_merges_sources_and_ignores_out_of_window():
    w = tally(
        [
            obs("Buteo jamaicensis", date(2026, 3, 9), source="inaturalist"),
            obs("Buteo jamaicensis", date(2026, 3, 12), source="ebird", common="Red-tailed Hawk"),
            obs("Buteo jamaicensis", date(2026, 1, 1)),
        ],
        *NOW,
    )
    t = w.species["Buteo jamaicensis"]
    assert t.n_observations == 2
    assert t.sources == ["inaturalist", "ebird"]
    assert t.common_name == "Red-tailed Hawk" and t.taxon_group == "bird"
    assert (t.first_seen, t.last_seen) == (date(2026, 3, 9), date(2026, 3, 12))
