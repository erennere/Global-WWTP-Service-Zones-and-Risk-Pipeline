"""Tests for hw_comparison (moved from the coverage-campaign files)."""
import numpy as np
import pandas as pd
import pytest
from wwtp.figures_scripts import composite_area_population_plots
from wwtp.pop_validation_scripts import hw_comparison, verification_script

pytestmark = pytest.mark.unit


def test_extract_voronoi_parameters_parses_path_components_and_filename_flags():
    path = "data/voronoi_layers/v3/lvl7/bf1800/logarithmic/appr_1_only_round_v3_lvl7_bf1800_mult.gpkg"

    params = hw_comparison.extract_voronoi_parameters(path)

    assert params["version"] == "3"
    assert params["level"] == 7
    assert params["buffer"] == 1800
    assert params["weight_type"] == "logarithmic"
    assert params["approach"] == "1"
    assert params["only_round"] is True
    assert params["weight_func"] == "_mult"


def test_ndvi_multiples_and_replace_inf_apply_expected_math():
    df = pd.DataFrame({"a": [10.0, 5.0], "b": [2.0, 0.0]})

    ndi = hw_comparison.ndvi(df, "a", "b", "ndi")
    mul = hw_comparison.multiples(df, "a", "b", "m")
    replaced = hw_comparison.replace_inf(pd.DataFrame({"m": [1.0, np.inf, -np.inf]}), "m")

    assert ndi["ndi"].tolist() == pytest.approx([(10 - 2) / (10 + 2 + 0.001), (5 - 0) / (5 + 0 + 0.001)])
    assert mul["m"].tolist() == pytest.approx([((10 - 2) / (2 + 0.001)) + 1, ((5 - 0) / (0 + 0.001)) + 1])
    assert replaced["m"].isna().sum() == 2
