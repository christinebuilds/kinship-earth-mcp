"""Spec 018: Taylor's power law check."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "taylors_power_law.py"
_spec = importlib.util.spec_from_file_location("taylors_power_law", _PATH)
tpl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tpl)


def _fit_species(sampler, mus, n=500):
    means, variances = [], []
    for mu in mus:
        x = sampler(mu, n)
        means.append(x.mean())
        variances.append(x.var(ddof=1))
    return tpl.fit_power_law(means, variances)[1]


def test_fitter_recovers_poisson_and_negative_binomial():
    rng = np.random.default_rng(42)
    mus = np.logspace(np.log10(2), np.log10(80), 30)
    b_pois = _fit_species(lambda mu, n: rng.poisson(mu, n), mus)
    assert abs(b_pois - 1.0) < 0.1

    def nb(mu, n):  # variance = mu ** 1.5 exactly
        k = mu ** 2 / (mu ** 1.5 - mu)
        return rng.negative_binomial(k, k / (k + mu), n)

    b_nb = _fit_species(nb, mus)
    assert abs(b_nb - 1.5) < 0.1


def test_aggregation_fills_zeros_and_sums_clusters():
    rows = [
        {"siteID": "S", "pointID": "p1", "taxonID": "AMRO", "clusterSize": "2"},
        {"siteID": "S", "pointID": "p1", "taxonID": "AMRO", "clusterSize": "1"},
        {"siteID": "S", "pointID": "p2", "taxonID": "AMRO", "clusterSize": "5"},
        {"siteID": "S", "pointID": "p2", "taxonID": "STJA", "clusterSize": "1"},
        {"siteID": "S", "pointID": "p3", "taxonID": "STJA", "clusterSize": "3"},
    ]
    counts = tpl.aggregate(rows)
    assert counts["AMRO"] == {("S", "p1"): 3.0, ("S", "p2"): 5.0, ("S", "p3"): 0.0}
    assert counts["STJA"] == {("S", "p1"): 0.0, ("S", "p2"): 1.0, ("S", "p3"): 3.0}
    mv = tpl.mean_var(counts, min_samples=3)
    assert mv["AMRO"][:2] == pytest.approx((8 / 3, np.var([3, 5, 0], ddof=1)))
    assert mv["STJA"][:2] == pytest.approx((4 / 3, np.var([0, 1, 3], ddof=1)))


@pytest.mark.network
def test_cached_snapshot_run_passes():
    sites = ["WREF", "HARV", "SRER", "OSBS", "KONZ", "CPER", "TALL", "BART", "UNDE", "JERC"]
    rows = tpl.cached_rows(sites, 2023)
    if not rows:
        pytest.skip("no cached NEON snapshot in canon/snapshots/neon/")
    result = tpl.run(rows, sites, 2023)
    assert result["pass"] is True
