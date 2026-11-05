"""The size and radius calibration recovers known relations and scores size classes equally."""
import numpy as np
import pandas as pd
import pytest

from src.calibration.service_radius import (
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
    return pd.DataFrame({"footprint_m2": footprint, "lagoon": lagoon, "sized": (footprint > 0) & ~lagoon,
                         "POP_SERVED": np.where(quality == 4, 1e7, people), "QUAL_POP": quality})


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
