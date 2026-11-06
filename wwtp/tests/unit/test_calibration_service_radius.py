"""The size and radius calibration recovers known relations and scores size classes equally."""

import numpy as np
import pandas as pd
import pytest

from wwtp.calibration.service_radius import (
    class_balanced_error,
    fit_footprint_to_people,
    summary,
)

pytestmark = pytest.mark.unit


def _plants(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    F = 10 ** rng.uniform(2, 5, n)
    people = 10 ** (1.5 + 0.7 * np.log10(F) + rng.normal(0, 0.2, n))
    lagoon = rng.random(n) < 0.1
    undetected = rng.random(n) < 0.1
    footprint = np.where(undetected, 0.0, F)
    people = np.where(lagoon | undetected, 2500.0, people)
    quality = np.where(rng.random(n) < 0.05, 4, 2)
    return pd.DataFrame(
        {
            "footprint_m2": footprint,
            "lagoon": lagoon,
            "sized": (footprint > 0) & ~lagoon,
            "POP_SERVED": np.where(quality == 4, 1e7, people),
            "QUAL_POP": quality,
        }
    )


def test_footprint_fit_recovers_the_relation_and_skips_11km_estimates():
    fit = fit_footprint_to_people(_plants(), ["POP_SERVED"], {"POP_SERVED": {"QUAL_POP": [4]}})
    assert fit["slope"] == pytest.approx(0.7, abs=0.02)
    assert fit["intercept"] == pytest.approx(1.5, abs=0.08)
    assert fit["unsized_people"] == 2500


def test_every_size_class_counts_equally():
    truth = np.array([1e3] * 90 + [1e6] * 10)
    small_right = np.where(truth < 1e4, truth, truth / 10)  # 90 % of plants right, all large ones 10x off
    large_right = np.where(truth < 1e4, truth * 2, truth)
    assert class_balanced_error(small_right, truth) == pytest.approx(0.5)
    assert class_balanced_error(large_right, truth) == pytest.approx(np.log10(2) / 2)
    assert np.median(np.abs(np.log10(small_right / truth))) == 0  # a plain median would prefer small_right


def test_plants_without_people_count_as_failures():
    truth = np.array([1e3, 1e3, 1e3])
    assert np.isinf(class_balanced_error(np.array([0.0, 0.0, 1e3]), truth))
    s = summary(np.array([0.0, 1e3, 1e3]), truth)
    assert s["within_factor_2"] == pytest.approx(2 / 3) and s["ratio <10k"] == pytest.approx(2 / 3)


def test_radius_choice_recovers_the_rule_of_a_synthetic_world(monkeypatch):
    """Plants whose real catchment is a disc of r10k * (S / 10,000) ** e km, holding
    exactly S people, with unsewered people around it: the census step must pick
    that e and r10k back from the grid."""
    import sys
    import types

    import geopandas as gpd
    import shapely
    from wwtp.calibration import service_radius as sr

    rng = np.random.default_rng(1)
    e_true, r10_true = 0.35, 3.75
    n = 36
    xs, ys = np.meshgrid(np.arange(6) * 120_000.0, np.arange(6) * 120_000.0)
    centres = np.column_stack([xs.ravel() - 1_500_000, ys.ravel() + 1_800_000])  # EPSG:5070, western US
    size = np.exp(rng.uniform(np.log(800), np.log(800_000), n))
    radius = r10_true * (size / 1e4) ** e_true * 1000
    pts, pop, polys = [], [], []
    for (x, y), s, r in zip(centres, size, radius):
        k = 400
        rr, th = r * np.sqrt(rng.uniform(0, 1, k)), rng.uniform(0, 2 * np.pi, k)
        pts.append(np.column_stack([x + rr * np.cos(th), y + rr * np.sin(th)]))
        pop.append(np.full(k, s / k))
        rr, th = rng.uniform(r * 1.05, r * 3, 300), rng.uniform(0, 2 * np.pi, 300)  # unsewered ring
        pts.append(np.column_stack([x + rr * np.cos(th), y + rr * np.sin(th)]))
        pop.append(np.full(300, s / 3000))
        polys.append(shapely.Point(x, y).buffer(r, 128))
    pts, pop = np.vstack(pts), np.concatenate(pop)
    ll = gpd.GeoSeries(gpd.points_from_xy(pts[:, 0], pts[:, 1]), crs=5070).to_crs(4326)
    bg = pd.DataFrame({"STATEFP": "49", "POPULATION": pop, "LONGITUDE": ll.x, "LATITUDE": ll.y})
    monkeypatch.setitem(sys.modules, "pyreadr", types.SimpleNamespace(read_r=lambda path: {"block_group2020": bg}))
    pl = gpd.GeoSeries(gpd.points_from_xy(centres[:, 0], centres[:, 1]), crs=5070).to_crs(4326)
    plants = pd.DataFrame({"lon": pl.x, "lat": pl.y})
    matches = pd.DataFrame({"plant": np.arange(n), "state": np.where(np.arange(n) % 2, "UT", "NV"), "poly": polys})

    res = sr.choose_radius(plants, size, matches, "unused.rda", 50_000, states={"UT": "49", "NV": "49"})
    assert (res["radius_exponent"], res["radius_at_10k_people_km"]) == (e_true, r10_true)
    assert res["in_sample"]["total_ratio"] == pytest.approx(1.0, abs=0.05)
    assert {f["held_out"] for f in res["leave_one_state_out"]} == {"NV", "UT"}


def test_load_plants_reads_footprint_lagoons_and_cwns(tmp_path):
    from wwtp.calibration.service_radius import load_plants

    path = tmp_path / "plants.csv"
    pd.DataFrame(
        {
            "lon": [8.0, 9.0, None],
            "lat": [50.0, 51.0, 52.0],
            "footprint_m2": [1000.0, None, 5.0],
            "lagoon_detected": ["False", "True", "False"],
            "CWNS_2022_ID": ["12345.0", None, "7"],
        }
    ).to_csv(path, index=False)
    df = load_plants(str(path))
    assert len(df) == 2 and df["sized"].tolist() == [True, False]
    assert df["cwns"].tolist()[0] == "00000012345" and pd.isna(df["cwns"].tolist()[1])
