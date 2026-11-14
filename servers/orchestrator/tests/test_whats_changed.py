"""ecology_whats_changed with faked adapters (offline, spec 022)."""

from datetime import datetime, timedelta, timezone

import pytest

from kinship_orchestrator import server
from kinship_shared.schema import EcologicalObservation, Location, Provenance, Quality, TaxonInfo


class FakeINat:
    id = "inaturalist"

    def __init__(self, per_window: dict[str, list[tuple[str, int]]]):
        # per_window: "now" | "before" | "last_year" -> [(species, count)]
        self.per_window = per_window
        self.calls = []

    async def search(self, params):
        self.calls.append(params)
        today = datetime.now(timezone.utc).date()
        start = datetime.fromisoformat(params.start_date).date()
        age = (today - start).days
        key = "now" if age < 20 else "before" if age < 60 else "last_year"
        day = start + timedelta(days=1)
        out, i = [], 0
        for name, count in self.per_window.get(key, []):
            for _ in range(count):
                i += 1
                out.append(EcologicalObservation(
                    id=f"inaturalist:{key}:{i}",
                    modality="occurrence",
                    location=Location(lat=params.lat, lng=params.lng),
                    observed_at=datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc),
                    taxon=TaxonInfo(scientific_name=name),
                    quality=Quality(tier=2),
                    provenance=Provenance(source_api="inaturalist", source_id=str(i), license="CC-BY-NC"),
                ))
        return out


@pytest.fixture
def fake(monkeypatch):
    def install(per_window):
        adapter = FakeINat(per_window)
        monkeypatch.setattr(server, "_inat", adapter)
        monkeypatch.setattr(server._ebird, "_api_key", None, raising=False)

        async def no_store(*a, **k):
            return None

        monkeypatch.setattr(server, "_store_turn", no_store)
        return adapter
    return install


async def test_arrivals_caveat_and_cap(fake):
    arrivals = [(f"Species number{i}", 2) for i in range(30)]
    adapter = fake({"now": arrivals + [("Calypte anna", 5)], "before": [("Calypte anna", 30)]})
    result = await server.ecology_whats_changed(lat=37.76123, lon=-122.42456)

    assert len(result["changes"]) == 25
    assert all(c["kind"] == "arrived" for c in result["changes"])
    assert result["summary"].startswith("30 arrivals, 0 departures")
    assert "volunteers" in result["caveat"] and "small" not in result["caveat"]
    assert result["location"] == {"lat": 37.76, "lon": -122.42, "radius_km": 5}
    assert result["observations"]["now"] == 65 and result["observations"]["last_year"] == 0
    assert result["sources_queried"] == ["inaturalist"]
    assert len(adapter.calls) == 3 and all(p.limit == 200 for p in adapter.calls)


async def test_small_sample_says_so(fake):
    fake({"now": [("Selasphorus sasin", 3)], "before": [("Zonotrichia atricapilla", 2)]})
    result = await server.ecology_whats_changed(lat=37.76, lon=-122.42)
    assert "sample here is small" in result["caveat"]
    assert [c["kind"] for c in result["changes"]] == ["arrived", "departed"]
    assert result["summary"] == "1 arrival, 1 departure in the last 14 days."
