import argparse
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from src import create_voronoi, starter


pytestmark = pytest.mark.unit


def test_normalize_optional_cli_value_preserves_explicit_empty_string():
    assert starter._normalize_optional_cli_value("   ", preserve_empty=True) == ""


def test_parse_config_overrides_from_argv_parses_all_supported_types():
    overrides = starter.parse_config_overrides(
        argv=[
            "prog",
            "--level", "9", "--version", "3", "--buffer", "12000",
            "--weight-method", "square_root", "--weight-func=",
            "--dynamic-buffering", "yes", "--size-measure", "footprint",
        ]
    )

    assert overrides == {
        "level": "9",
        "version": "3",
        "buffer": 12000,
        "weight_method": "square_root",
        "weight_func": "",
        "dynamic_buffering": True,
        "size_measure": "footprint",
    }


def test_parse_config_overrides_from_namespace_normalizes_optional_values():
    args = Namespace(
        level=" 7 ",
        version=" null ",
        buffer=None,
        weight_method="logarithmic",
        weight_func="add",
        dynamic_buffering="false",
        size_measure=" reported ",
    )

    overrides = starter.parse_config_overrides(args=args)

    assert overrides == {
        "level": "7",
        "version": None,
        "buffer": None,
        "weight_method": "logarithmic",
        "weight_func": "add",
        "dynamic_buffering": False,
        "size_measure": "reported",
    }


def test_add_standard_override_arguments_parses_named_flags():
    parser = argparse.ArgumentParser()
    starter.add_standard_override_arguments(parser)
    args = parser.parse_args([
        "--level", "7",
        "--version", "3",
        "--buffer", "1200",
        "--weight-method", "logarithmic",
        "--weight-func", "mult",
        "--dynamic-buffering", "true",
        "--size-measure", "footprint",
    ])

    overrides = starter.parse_config_overrides(args=args)

    assert overrides == {
        "level": "7",
        "version": "3",
        "buffer": 1200,
        "weight_method": "logarithmic",
        "weight_func": "mult",
        "dynamic_buffering": True,
        "size_measure": "footprint",
    }


def test_parse_config_overrides_ignores_positional_tokens():
    """Positional overrides were removed: a stray token must not become `level`."""
    overrides = starter.parse_config_overrides(
        argv=["prog", "required_arg", "8", "4", "1500", "linear", "mult", "0", "1.5"],
    )

    assert overrides == {
        "level": None,
        "version": None,
        "buffer": None,
        "weight_method": None,
        "weight_func": None,
        "dynamic_buffering": None,
        "size_measure": None,
    }


def test_parse_config_overrides_reads_named_flags_from_argv():
    overrides = starter.parse_config_overrides(
        argv=[
            "prog", "positional-arg",
            "--level", "8", "--version", "4", "--buffer", "1500",
            "--weight-method", "linear", "--weight-func", "mult",
            "--dynamic-buffering", "0", "--size-measure", "reported",
        ],
    )

    assert overrides == {
        "level": "8",
        "version": "4",
        "buffer": 1500,
        "weight_method": "linear",
        "weight_func": "mult",
        "dynamic_buffering": False,
        "size_measure": "reported",
    }


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["prog", "--buffer", "not-an-int"], "Invalid buffer"),
        (["prog", "--weight-func", "bad-mode"], "Invalid weight_func"),
        (["prog", "--dynamic-buffering", "maybe"], "Invalid dynamic_buffering"),
        (["prog", "--size-measure", "capacity"], "Invalid size_measure"),
    ],
)
def test_parse_config_overrides_rejects_invalid_optional_values(argv, message):
    with pytest.raises(ValueError, match=message):
        starter.parse_config_overrides(argv=argv)


def test_normalize_cfg_path_handles_relative_absolute_and_url_inputs(tmp_path):
    base_dir = tmp_path / "config_root"
    base_dir.mkdir()

    relative_result = starter._normalize_cfg_path("data/example.csv", str(base_dir))
    absolute_input = str((tmp_path / "absolute.csv").resolve())

    assert relative_result == str((base_dir / "data" / "example.csv").resolve())
    assert starter._normalize_cfg_path(absolute_input, str(base_dir)) == absolute_input
    assert starter._normalize_cfg_path("s3://bucket/key/file.parquet", str(base_dir)) == "s3://bucket/key/file.parquet"
    assert starter._normalize_cfg_path(42, str(base_dir)) == 42


def test_load_config_expands_relative_paths_and_aliases(write_test_config):
    config_path = write_test_config(
        {
            "combine_watersheds": {
                "level": "5",
            },
            "correct_locations_w_OSM": {
                "version": "9",
            },
            "create_voronoi": {
                "buffer": 1500,
                "weight_method": "square_root",
                "weight_func": "add",
                "dynamic_buffering": False,
            },
        }
    )

    cfg = starter.load_config(script_name="create_voronoi", config=str(config_path))

    assert cfg["level"] == "5"
    assert cfg["version"] == "9"
    assert cfg["weight_type"] == "sq"
    assert cfg["weight_func"] == "add"
    assert cfg["weight_func_suffix"] == "_add"
    assert cfg["buffer_path_token"] == "1500"
    assert cfg["distance_fn"] is create_voronoi.default_distance_additive
    assert cfg["paths"]["bboxes"] == str((config_path.parent / "data" / "bboxes.csv").resolve())
    assert "bf1500" in cfg["paths"]["voronoi_dir"]


def test_load_config_dynamic_buffering_uses_size_measure_token_and_empty_weight_func(write_test_config):
    config_path = write_test_config(
        {
            "create_voronoi": {
                "buffer": 2200,
                "weight_method": "logarithmic",
                "weight_func": "mult",
                "dynamic_buffering": True,
                "size_measure": "reported",
            },
        }
    )

    cfg = starter.load_config(
        script_name="create_voronoi",
        config=str(config_path),
        weight_func="",
        dynamic_buffering=True,
        size_measure="footprint",
    )

    assert cfg["weight_func"] == ""
    assert cfg["weight_func_suffix"] == ""
    assert cfg["buffer_path_token"] == "fp"
    assert cfg["dynamic_buffering"] is True
    assert cfg["size_measure"] == "footprint"
    assert cfg["distance_fn"] is create_voronoi.default_distance_multiplicative


SIZE_CONFIG = {
    "size_measure": "reported",
    "footprint_to_people": {"intercept": 1.0, "slope": 0.8, "unsized_people": 2000},
    "reported_size_columns": ["POP_SERVED"],
    "reported_size_skip": {"POP_SERVED": {"QUAL_POP": [4]}},
}
SIZE_KWARGS = {
    "size_measure": "reported",
    "footprint_to_people": {"intercept": 1.0, "slope": 0.8, "unsized_people": 2000.0},
    "reported_size_columns": ["POP_SERVED"],
    "reported_size_skip": {"POP_SERVED": {"QUAL_POP": [4]}},
}


def test_load_config_derives_calculate_buffer_kwargs_with_expected_keys(write_test_config):
    config_path = write_test_config(
        {
            "create_voronoi": {
                "buffer": 9000,
                "weight_method": "size",
                "weight_func": "mult",
                "dynamic_buffering": True,
                "max_buffer": 50000,
                **SIZE_CONFIG,
                "radius_calibration": {"reported": {"radius_exponent": 0.3, "radius_at_10k_people_km": 4.0}},
            },
        }
    )

    cfg = starter.load_config(script_name="create_voronoi", config=str(config_path))

    assert cfg["calculate_buffer_kwargs"] == {
        "buffer": 9000,
        "dynamic_buffering": True,
        "max_buffer": 50000,
        **SIZE_KWARGS,
        "radius_exponent": 0.3,
        "radius_at_10k_people_km": 4.0,
    }


def test_fixed_buffering_needs_no_radius_calibration(write_test_config):
    config_path = write_test_config(
        {"create_voronoi": {"buffer": 9000, "dynamic_buffering": False, "max_buffer": 50000, **SIZE_CONFIG,
                            "radius_calibration": {}}}
    )
    cfg = starter.load_config(script_name="create_voronoi", config=str(config_path))
    # the plant size stays available for weight_method 'size'
    assert cfg["calculate_buffer_kwargs"] == {"buffer": 9000, "dynamic_buffering": False, "max_buffer": 50000, **SIZE_KWARGS}
    assert cfg["buffer_path_token"] == "9000"


def test_size_measure_without_calibration_entry_is_rejected():
    table = {"footprint": {"radius_exponent": 0.4, "radius_at_10k_people_km": 4.0}}
    assert starter._calibration_row(table, "footprint")["radius_exponent"] == 0.4
    with pytest.raises(ValueError, match="no radius_calibration entry"):
        starter._calibration_row(table, "reported")
    with pytest.raises(ValueError, match="must have"):
        starter._calibration_row({"reported": {"radius_exponent": 0.4}}, "reported")


def test_load_config_normalizes_weight_func_from_yaml(write_test_config):
    config_path = write_test_config(
        {
            "create_voronoi": {
                "buffer": 2200,
                "weight_method": "logarithmic",
                "weight_func": " add ",
                "dynamic_buffering": False,
            },
        }
    )

    cfg = starter.load_config(script_name="create_voronoi", config=str(config_path))

    assert cfg["weight_func"] == "add"
    assert cfg["weight_func_suffix"] == "_add"
    assert cfg["distance_fn"] is create_voronoi.default_distance_additive


def test_load_config_rejects_invalid_weight_method(write_test_config):
    config_path = write_test_config(
        {
            "create_voronoi": {
                "weight_method": "banana",
            }
        }
    )

    with pytest.raises(ValueError, match="Invalid weight_method"):
        starter.load_config(script_name="create_voronoi", config=str(config_path))


def test_load_config_allows_sections_without_unrelated_voronoi_runtime_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "correct_locations_w_OSM": {
                    "version": "2",
                    "rad": 5000,
                    "paths": {
                        "data_dir": "./data",
                        "paul_corrected_filepath": "{data_dir}/Enhanced_HW_WWTP__jun20_2025.geojson",
                        "corrected_south": "{data_dir}/corrected_WWTP_enhanced_v{version}.geojson",
                        "corrected_all_filepath": "{data_dir}/all_merged_v{version}.gpkg",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = starter.load_config(script_name="correct_locations_w_OSM", config=str(config_path))

    assert cfg["rad"] == 5000
    assert cfg["version"] == "2"
    assert cfg["paths"]["corrected_all_filepath"] == str((tmp_path / "data" / "all_merged_v2.gpkg").resolve())
    assert "buffer" not in cfg
    assert "weight_type" not in cfg


def test_resolve_config_rejects_missing_script_section():
    with pytest.raises(starter.ConfigResolutionError, match="missing"):
        starter.resolve_config("missing_script", {"correct_locations_w_OSM": {}})


def test_resolve_config_rejects_non_mapping_script_section():
    with pytest.raises(starter.ConfigResolutionError, match="must be a mapping"):
        starter.resolve_config("download_pop", {"download_pop": []})


def test_resolve_config_uses_yaml_section_order():
    raw_config = {
        "combine_watersheds": {"shared": "wrong"},
        "final_data_merge": {"shared": "right"},
        "create_voronoi": {"shared": None},
    }

    resolved = starter.resolve_config("create_voronoi", raw_config)

    assert resolved["shared"] == "wrong"


def test_resolve_config_rejects_future_section_fallbacks():
    raw_config = {
        "correct_locations_w_OSM": {"shared": None},
        "create_voronoi": {"shared": "future-only"},
    }

    with pytest.raises(starter.ConfigResolutionError, match="shared"):
        starter.resolve_config("correct_locations_w_OSM", raw_config)


def test_resolve_config_rejects_non_mapping_earlier_section():
    raw_config = {
        "correct_locations_w_OSM": [],
        "download_pop": {"zoom_level": 8},
    }

    with pytest.raises(starter.ConfigResolutionError, match="must be a mapping"):
        starter.resolve_config("download_pop", raw_config)


def test_resolve_config_rejects_unresolved_null_key():
    raw_config = {
        "correct_locations_w_OSM": {
            "paths": {
                "data_dir": "./data",
            }
        },
        "download_pop": {
            "paths": None,
            "zoom_level": None,
        },
    }

    with pytest.raises(starter.ConfigResolutionError, match="zoom_level"):
        starter.resolve_config("download_pop", raw_config)


def test_resolve_config_limits_nested_inheritance_to_declared_keys():
    raw_config = {
        "correct_locations_w_OSM": {
            "paths": {
                "data_dir": "./data",
                "corrected_south": "./data/corrected.geojson",
            }
        },
        "combine_watersheds": {
            "paths": {
                "data_dir": None,
                "watershed": "{data_dir}/watershed.gpkg",
            }
        },
    }

    resolved = starter.resolve_config("combine_watersheds", raw_config)

    assert resolved["paths"] == {
        "data_dir": "./data",
        "watershed": "{data_dir}/watershed.gpkg",
    }