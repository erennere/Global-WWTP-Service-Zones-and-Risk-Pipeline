"""The radius calibration recovers a known size-reach relation."""
import numpy as np
import pandas as pd
import pytest

from src.calibration.service_radius import (
    coverage,
    fit_shape,
    two_neighbour_distance_km,
)

pytestmark = pytest.mark.unit


def _matches(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    F = 10 ** rng.uniform(2, 5, n)
    r95 = 0.5 * F ** 0.3 * 10 ** rng.normal(0, 0.2, n)  # km
    sized = rng.random(n) > 0.2
    return pd.DataFrame({"footprint_m2": np.where(sized, F, 0.0), "sized": sized, "r95_km": r95,
                         "nnd2_km": 10 ** rng.uniform(0, 1.5, n)})


def test_fit_recovers_exponent_and_ignores_unrelated_neighbour_distance():
    shape = fit_shape(_matches())
    assert shape["exponent"] == pytest.approx(0.3, abs=0.02)
    assert shape["r2_neighbour_distance"] < 0.01


def test_each_row_covers_its_quantile():
    m = _matches()
    shape = fit_shape(m)
    cov = coverage(m[m["sized"]], shape)
    for q, c in cov.items():
        assert c == pytest.approx(q, abs=0.03)
    assert list(cov.values()) == sorted(cov.values())


def test_two_neighbour_distance_is_in_kilometres():
    lon = np.array([8.0, 8.0, 8.0])
    lat = np.array([50.0, 50.0 + 1 / 111.195, 50.0 + 3 / 111.195])  # 1 km and 3 km apart
    d = two_neighbour_distance_km(lon, lat)
    assert d[0] == pytest.approx(2.0, abs=0.01)  # neighbours at 1 and 3 km
