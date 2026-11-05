import pytest
import pandas as pd

from src.sensitivity_analysis_scripts.compare_pop_sweep_hw_eu import build_alias_order, parse_pop_output_path


pytestmark = pytest.mark.unit


def test_parse_pop_output_path_accepts_integer_buffer_token():
    filepath = (
        "/tmp/pop_voronoi_layers/v2/lvl7/bf1500/log/"
        "pop_added_appr_1_v2_lvl7_bf1500_log_mult.gpkg"
    )

    parsed = parse_pop_output_path(filepath)

    assert parsed is not None
    assert parsed["version"] == "2"
    assert parsed["level"] == 7
    assert parsed["buffer"] == "1500"
    assert parsed["weight_type"] == "log"
    assert parsed["approach"] == "1"
    assert parsed["only_round"] is False
    assert parsed["weight_func"] == "_log_mult"


def test_parse_pop_output_path_accepts_dynamic_k_buffer_token():
    filepath = (
        "/tmp/pop_voronoi_layers/v2/lvl7/bfk0_75/log/"
        "pop_added_appr_1_only_round_v2_lvl7_bfk0_75_log.gpkg"
    )

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
    filepath = (
        "/tmp/pop_voronoi_layers/v2/lvl7/bfk0.75/log/"
        "pop_added_appr_1_v2_lvl7_bfk0.75_log_add.gpkg"
    )

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
    filepath = (
        "/tmp/pop_voronoi_layers/v2/lvl7/bfk0_75/log/"
        "pop_added_appr_1_v2_lvl7_bf1500_log_mult.gpkg"
    )

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



def test_parse_pop_output_path_accepts_radius_quantile_buffer_token():
    parsed = parse_pop_output_path(
        "/tmp/pop_voronoi_layers/v2/lvl8/bfq0_6/log/pop_added_appr_1_v2_lvl8_bfq0_6_log_mult.gpkg"
    )
    assert parsed is not None and parsed["buffer"] == "q0_6"
