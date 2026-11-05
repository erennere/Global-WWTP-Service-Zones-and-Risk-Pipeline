from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.figures_scripts import composite_area_population_plots
from src.pop_validation_scripts import hw_comparison, verification_script


pytestmark = pytest.mark.unit


def test_find_verification_watersheds_marks_single_and_verification_flags():
    df = pd.DataFrame(
        {
            "HYBAS_ID": [101, 101, 202],
            "total_area": [10.0, 0.0, 5.0],
        }
    )

    result = verification_script.find_verification_watersheds(df, percent_verification=0.5)

    assert result["is_single_points"].tolist() == [False, False, True]
    assert result["use_verify"].tolist() == [True, False, False]
    assert result["watersheds_chosen"].tolist() == [True, True, False]


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


def test_resolve_zonal_sum_column_prefers_requested_or_latest_year():
    df = pd.DataFrame({"2019_zonal_sum": [1], "2024_zonal_sum": [2], "foo": [3]})

    preferred = composite_area_population_plots.resolve_zonal_sum_column(df, "2019_zonal_sum")
    fallback = composite_area_population_plots.resolve_zonal_sum_column(df, "missing_col")

    assert preferred == "2019_zonal_sum"
    assert fallback == "2024_zonal_sum"



def test_resolve_zonal_sum_column_raises_when_no_candidates_exist():
    with pytest.raises(KeyError, match=r"No '\*_zonal_sum' column"):
        composite_area_population_plots.resolve_zonal_sum_column(pd.DataFrame({"x": [1]}), "2024_zonal_sum")


def test_clip_outliers_removes_inf_nan_and_quantile_outliers():
    series = pd.Series([1, 2, 3, 1000, np.inf, -np.inf, np.nan])

    clipped = composite_area_population_plots.clip_outliers(series, lower_q=0.05, upper_q=0.95)

    assert clipped.notna().all()
    assert np.isfinite(clipped).all()
    assert 1000 not in clipped.tolist()


def test_clip_outliers_rejects_invalid_quantile_bounds():
    with pytest.raises(ValueError, match="Quantile bounds"):
        composite_area_population_plots.clip_outliers(pd.Series([1, 2, 3]), lower_q=0.9, upper_q=0.1)


def test_make_category_color_map_is_deterministic_for_unique_categories():
    values = ["B", "A", None, "A"]

    color_map = composite_area_population_plots.make_category_color_map(values)

    assert set(color_map.keys()) == {"A", "B", "Unknown"}
    assert color_map["A"] == composite_area_population_plots.make_category_color_map(values)["A"]


def test_add_one_to_one_line_adds_reference_line_for_non_empty_data():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x_vals = pd.Series([1.0, 2.0, 3.0])
    y_vals = pd.Series([1.5, 2.5, 3.5])

    composite_area_population_plots.add_one_to_one_line(ax, x_vals, y_vals)

    assert len(ax.lines) == 1
    x_line, y_line = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    assert list(x_line) == list(y_line)
    plt.close(fig)


def test_resolve_zonal_sum_column_ignores_unparseable_candidates():
    df = pd.DataFrame({"abc_zonal_sum": [1], "2024_zonal_sum": [2]})

    result = composite_area_population_plots.resolve_zonal_sum_column(df, "missing_col")

    assert result == "2024_zonal_sum"


def test_add_one_to_one_line_skips_empty_series_and_handles_flat_range():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, (ax_empty, ax_flat) = plt.subplots(1, 2)

    composite_area_population_plots.add_one_to_one_line(ax_empty, pd.Series([np.nan]), pd.Series([np.nan]))
    composite_area_population_plots.add_one_to_one_line(ax_flat, pd.Series([5.0]), pd.Series([5.0]))

    assert len(ax_empty.lines) == 0
    assert len(ax_flat.lines) == 1
    x_line = ax_flat.lines[0].get_xdata()
    assert x_line[1] > x_line[0]
    plt.close(fig)
