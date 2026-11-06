import pytest
import pandas as pd

from wwtp.sensitivity_analysis_scripts.compare_pop_sweep_hw_eu import build_alias_order, parse_pop_output_path


pytestmark = pytest.mark.unit


def test_parse_pop_output_path_accepts_integer_buffer_token():
    filepath = "/tmp/pop_voronoi_layers/v2/lvl7/bf1500/log/pop_added_appr_1_v2_lvl7_bf1500_log_mult.gpkg"

    parsed = parse_pop_output_path(filepath)

    assert parsed is not None
    assert parsed["version"] == "2"
    assert parsed["level"] == 7
    assert parsed["buffer"] == "1500"
    assert parsed["weight_type"] == "log"
    assert parsed["approach"] == "1"
    assert parsed["only_round"] is False
    assert parsed["weight_func"] == "_log_mult"
    assert parsed["distance_weighting"] == "mult"


def test_parse_pop_output_path_accepts_dynamic_k_buffer_token():
    filepath = "/tmp/pop_voronoi_layers/v2/lvl7/bfk0_75/log/pop_added_appr_1_only_round_v2_lvl7_bfk0_75_log.gpkg"

    parsed = parse_pop_output_path(filepath)

    assert parsed is not None
    assert parsed["version"] == "2"
    assert parsed["level"] == 7
    assert parsed["buffer"] == "k0_75"
    assert parsed["weight_type"] == "log"
    assert parsed["approach"] == "1"
    assert parsed["only_round"] is True
    assert parsed["weight_func"] == "_log"


def test_parse_pop_output_path_accepts_dotted_dynamic_k_buffer_token():
    filepath = "/tmp/pop_voronoi_layers/v2/lvl7/bfk0.75/log/pop_added_appr_1_v2_lvl7_bfk0.75_log_add.gpkg"

    parsed = parse_pop_output_path(filepath)

    assert parsed is not None
    assert parsed["buffer"] == "k0.75"
    assert parsed["weight_type"] == "log"
    assert parsed["weight_func"] == "_log_add"


def test_parse_pop_output_path_accepts_without_weight_type_subdir():
    filepath = "/tmp/pop_voronoi_layers/v2/lvl7/bf1500/pop_added_appr_1_v2_lvl7_bf1500_log_mult.gpkg"

    parsed = parse_pop_output_path(filepath)

    assert parsed is not None
    assert parsed["buffer"] == "1500"
    assert parsed["weight_type"] == ""


def test_parse_pop_output_path_rejects_buffer_token_mismatch():
    filepath = "/tmp/pop_voronoi_layers/v2/lvl7/bfk0_75/log/pop_added_appr_1_v2_lvl7_bf1500_log_mult.gpkg"

    assert parse_pop_output_path(filepath) is None


def test_build_alias_order_can_prioritize_hw_source():
    summary = pd.DataFrame(
        [
            {"alias": "A", "source": "HW", "sensitivity_score": 1.0},
            {"alias": "A", "source": "EU", "sensitivity_score": 9.0},
            {"alias": "B", "source": "HW", "sensitivity_score": 4.0},
            {"alias": "B", "source": "EU", "sensitivity_score": 2.0},
        ]
    )

    unweighted_order, _ = build_alias_order(summary, hw_weight=0.5)
    hw_weighted_order, score_table = build_alias_order(summary, hw_weight=0.8)

    assert unweighted_order[0] == "B"
    assert hw_weighted_order[0] == "A"
    assert score_table["hw_weight_used"].iloc[0] == pytest.approx(0.8)
    assert score_table["eu_weight_used"].iloc[0] == pytest.approx(0.2)


def test_parse_pop_output_path_accepts_size_measure_buffer_token():
    parsed = parse_pop_output_path(
        "/tmp/pop_voronoi_layers/v2/lvl8/bfq0_6/log/pop_added_appr_1_v2_lvl8_bfq0_6_log_mult.gpkg"
    )
    assert parsed is not None and parsed["buffer"] == "q0_6"


# --- metrics against hand-computed values ------------------------------------
import numpy as np

from wwtp.sensitivity_analysis_scripts import compare_pop_sweep_hw_eu as cmp


def test_identical_predictions_score_perfectly():
    df = pd.DataFrame({"pred": [10.0, 200.0, 3000.0, 45.0], "ref": [10.0, 200.0, 3000.0, 45.0]})
    m = cmp.compute_sensitivity_metrics(df, "pred", "ref")
    assert m["n"] == 4 and m["median_abs_ndi"] == pytest.approx(0) and m["median_alpha"] == pytest.approx(1)
    assert m["rmse_log_ratio"] == pytest.approx(0, abs=1e-9) and m["spearman_r"] == pytest.approx(1)


def test_doubling_every_prediction_gives_alpha_2_and_log_ratio_ln2():
    df = pd.DataFrame({"pred": [20.0, 400.0, 6000.0], "ref": [10.0, 200.0, 3000.0]})
    m = cmp.compute_sensitivity_metrics(df, "pred", "ref")
    assert m["median_alpha"] == pytest.approx(2, rel=1e-3)
    assert m["rmse_log_ratio"] == pytest.approx(np.log(2), rel=1e-3)


@pytest.mark.parametrize("frame", [pd.DataFrame({"x": [1.0]}), pd.DataFrame({"pred": [np.inf], "ref": [np.nan]})])
def test_unusable_inputs_give_n_0(frame):
    assert cmp.compute_sensitivity_metrics(frame, "pred", "ref")["n"] == 0


def test_zscore_of_a_constant_is_zero():
    assert cmp._safe_zscore(pd.Series([3.0, 3.0, 3.0])).tolist() == [0.0, 0.0, 0.0]
    z = cmp._safe_zscore(pd.Series([1.0, 2.0, 3.0]))
    assert z.mean() == pytest.approx(0) and z.std(ddof=0) == pytest.approx(1)


def test_alias_order_uses_the_available_scores_only():
    summary = pd.DataFrame(
        {"alias": ["A", "A", "B", "C"], "source": ["HW", "EU", "HW", "EU"], "sensitivity_score": [1.0, 3.0, 2.5, 0.5]}
    )
    order, table = cmp.build_alias_order(summary, hw_weight=0.5)
    scores = table.set_index("alias")["joint_score"]
    assert scores["A"] == pytest.approx(2.0) and scores["B"] == pytest.approx(2.5) and scores["C"] == pytest.approx(0.5)
    assert order == ["C", "A", "B"]
    with pytest.raises(ValueError):
        cmp.build_alias_order(summary, hw_weight=-0.1)
    with pytest.raises(ValueError):
        cmp.build_alias_order(summary, hw_weight=0.0, eu_weight=0.0)


def test_listing_skips_temp_files_and_duplicate_unweighted_runs(tmp_path):
    base = tmp_path / "pop_voronoi_layers" / "v2" / "lvl6" / "bfrep"
    files = [
        base / "sz" / "pop_added_appr_1_v2_lvl6_bfrep_sz_mult.gpkg",
        base / "li" / "pop_added_appr_1_v2_lvl6_bfrep_li.gpkg",
        base / "log" / "pop_added_appr_1_v2_lvl6_bfrep_log.gpkg",  # same run without weighting
        base / "sz" / "temp_pop_added_appr_1_v2_lvl6_bfrep_sz_mult.gpkg",
        base / "sz" / "notes.gpkg",
    ]
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("")
    records = cmp.make_aliases(cmp.list_pop_output_files(tmp_path))
    assert len(records) == 2 and [r["alias"] for r in records] == ["F001", "F002"]
    assert {r["buffer"] for r in records} == {"rep"}
