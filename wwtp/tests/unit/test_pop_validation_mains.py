from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from wwtp.pop_validation_scripts import eu_comparison, hw_comparison, verification_script


pytestmark = pytest.mark.unit


def test_verification_main_writes_ver_unver_and_single_outputs(monkeypatch, tmp_path):
    verification_dir = str(tmp_path / "verification")
    pop_dir = str(tmp_path / "pop")
    cfg = {
        "paths": {
            "verification_dir": verification_dir,
            "pop_output_dir": pop_dir,
        },
        "percent_verification": 0.5,
    }
    captured = {"writes": []}

    gdf = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 101, 202, 303, 303],
            "total_area": [10.0, 0.0, 5.0, 0.0, 0.0],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(verification_script.os, "chdir", lambda path: None)
    monkeypatch.setattr(verification_script, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(verification_script, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(verification_script.os.path, "isdir", lambda path: path == pop_dir)
    monkeypatch.setattr(verification_script.os.path, "exists", lambda path: False)
    monkeypatch.setattr(verification_script.os, "makedirs", lambda path, exist_ok=False: captured.setdefault("makedirs", (path, exist_ok)))
    monkeypatch.setattr(verification_script.os, "listdir", lambda path: ["sample.gpkg"])
    monkeypatch.setattr(verification_script.gpd, "read_file", lambda path: gdf.copy())
    monkeypatch.setattr(
        verification_script,
        "ensure_output_dir_for_file",
        lambda path: captured.setdefault("ensured", []).append(path),
    )

    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, path, driver=None, index=None, **kwargs):
        captured["writes"].append({"path": path, "driver": driver, "index": index, "rows": len(self)})

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        verification_script.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert captured["makedirs"] == (verification_dir, True)
    write_paths = [w["path"] for w in captured["writes"]]
    assert os.path.join(verification_dir, "ver_sample.gpkg") in write_paths
    assert os.path.join(verification_dir, "unver_sample.gpkg") in write_paths
    assert os.path.join(verification_dir, "single_sample.gpkg") in write_paths
    assert all(w["driver"] == "GPKG" and w["index"] is False for w in captured["writes"])


def test_verification_main_requires_existing_pop_output_dir(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "verification_dir": str(tmp_path / "verification"),
            "pop_output_dir": str(tmp_path / "missing_pop"),
        },
        "percent_verification": 0.5,
    }

    monkeypatch.setattr(verification_script.os, "chdir", lambda path: None)
    monkeypatch.setattr(verification_script, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(verification_script, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(verification_script.os.path, "isdir", lambda path: False)

    with pytest.raises(FileNotFoundError, match="Population output directory"):
        verification_script.main()


def test_verification_main_processes_only_gpkg_files(monkeypatch, tmp_path):
    verification_dir = str(tmp_path / "verification")
    pop_dir = str(tmp_path / "pop")
    cfg = {
        "paths": {
            "verification_dir": verification_dir,
            "pop_output_dir": pop_dir,
        },
        "percent_verification": 0.5,
    }
    captured = {"read_paths": []}

    gdf = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [1],
            "total_area": [1.0],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(verification_script.os, "chdir", lambda path: None)
    monkeypatch.setattr(verification_script, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(verification_script, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(verification_script.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(verification_script.os.path, "exists", lambda path: True)
    monkeypatch.setattr(verification_script.os, "listdir", lambda path: ["a.gpkg", "notes.txt"])
    monkeypatch.setattr(
        verification_script.gpd,
        "read_file",
        lambda path: captured["read_paths"].append(path) or gdf.copy(),
    )
    monkeypatch.setattr(verification_script, "ensure_output_dir_for_file", lambda path: None)

    original_to_file = gpd.GeoDataFrame.to_file
    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, path, driver=None, index=None, **kwargs: None)
        verification_script.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert captured["read_paths"] == [os.path.join(pop_dir, "a.gpkg")]


def test_find_verification_watersheds_rejects_invalid_threshold():
    gdf = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 101],
            "total_area": [10.0, 0.0],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="percent_verification"):
        verification_script.find_verification_watersheds(gdf, 1.5)


def test_find_verification_watersheds_requires_columns():
    gdf_missing_total_area = gpd.GeoDataFrame(
        {"HYBAS_ID": [101], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    with pytest.raises(KeyError, match="total_area"):
        verification_script.find_verification_watersheds(gdf_missing_total_area, 0.5)

    gdf_missing_hybas = gpd.GeoDataFrame(
        {"total_area": [1.0], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    with pytest.raises(KeyError, match="HYBAS_ID"):
        verification_script.find_verification_watersheds(gdf_missing_hybas, 0.5)


def test_hw_comparison_main_dispatches_only_gpkg_files(monkeypatch, tmp_path):
    ver_dir = str(tmp_path / "verification")
    plots_dir = str(tmp_path / "plots")
    cfg = {
        "paths": {
            "verification_dir": ver_dir,
            "hw_plots_dir": plots_dir,
        }
    }
    captured = []

    gdf = gpd.GeoDataFrame(
        {
            "POP_SERVED": [100.0],
            "2020_zonal_sum": [110.0],
            "QUAL_POP": ["1.0"],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(hw_comparison.os, "chdir", lambda path: None)
    monkeypatch.setattr(hw_comparison, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(hw_comparison, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(hw_comparison.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(hw_comparison.os, "listdir", lambda path: ["a.gpkg", "notes.txt"])
    monkeypatch.setattr(hw_comparison.gpd, "read_file", lambda path: gdf.copy())
    monkeypatch.setattr(hw_comparison, "extract_voronoi_parameters", lambda filepath: {"approach": "1"})
    monkeypatch.setattr(
        hw_comparison,
        "orchestrate_single",
        lambda gdf, approach, plot_args, output_dir, filename, pop_col='POP_SERVED': captured.append(
            {
                "approach": approach,
                "output_dir": output_dir,
                "filename": filename,
                "pop_col": pop_col,
                "rows": len(gdf),
                "save": plot_args["save"],
            }
        ),
    )

    hw_comparison.main()

    assert captured == [
        {
            "approach": "1",
            "output_dir": plots_dir,
            "filename": "a.gpkg",
            "pop_col": "POP_SERVED",
            "rows": 1,
            "save": True,
        }
    ]


def test_hw_comparison_main_returns_when_verification_dir_missing(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "verification_dir": str(tmp_path / "missing_verification"),
            "hw_plots_dir": str(tmp_path / "plots"),
        }
    }

    monkeypatch.setattr(hw_comparison.os, "chdir", lambda path: None)
    monkeypatch.setattr(hw_comparison, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(hw_comparison, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(hw_comparison.os.path, "isdir", lambda path: False)
    monkeypatch.setattr(
        hw_comparison.gpd,
        "read_file",
        lambda path: (_ for _ in ()).throw(AssertionError("read_file should not run when verification dir is missing")),
    )

    hw_comparison.main()


def test_eu_comparison_main_assigns_nearest_filters_and_dispatches(monkeypatch, tmp_path):
    ver_dir = str(tmp_path / "verification")
    os.makedirs(ver_dir, exist_ok=True)
    plots_dir = str(tmp_path / "plots")
    ref_path = str(tmp_path / "eu_ref.gpkg")
    cfg = {
        "paths": {
            "verification_dir": ver_dir,
            "eu_plots_dir": plots_dir,
            "eu_ref_filepath": ref_path,
        },
        "threshold": 500,
        "eu_reference_factor": 1,
    }
    captured = {}

    ref_gdf = gpd.GeoDataFrame(
        {
            "uwwCapacity": [10.0],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    input_gdf = gpd.GeoDataFrame(
        {
            "2020_zonal_sum": [100.0, 200.0],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(eu_comparison.os, "chdir", lambda path: None)
    monkeypatch.setattr(eu_comparison, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(eu_comparison, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(eu_comparison.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(eu_comparison.os, "listdir", lambda path: ["eu_case.gpkg", "ignore.txt"])

    def fake_read_file(path):
        if path == ref_path:
            return ref_gdf.copy()
        return input_gdf.copy()

    monkeypatch.setattr(eu_comparison.gpd, "read_file", fake_read_file)
    monkeypatch.setattr(eu_comparison, "extract_voronoi_parameters", lambda filepath: {"approach": "2"})

    def fake_assign_to_nearest(gdf, ref_file, threshold, one_to_one=False):
        captured["assign"] = {
            "threshold": threshold, "ref_cols": ref_file.columns.tolist(), "rows": len(gdf), "one_to_one": one_to_one,
        }
        return gdf.assign(uwwCapacity=[1.0, pd.NA], POP_SERVED_EU=[10.0, pd.NA])

    monkeypatch.setattr(eu_comparison, "assign_to_nearest", fake_assign_to_nearest)
    monkeypatch.setattr(
        eu_comparison,
        "orchestrate_single",
        lambda gdf, approach, plot_args, output_dir, filename, pop_col='POP_SERVED', **kwargs: captured.setdefault(
            "orchestrate",
            {
                "rows": len(gdf),
                "approach": approach,
                "output_dir": output_dir,
                "filename": filename,
                "pop_col": pop_col,
                "save": plot_args["save"],
            },
        ),
    )

    eu_comparison.main()

    assert captured["assign"] == {"threshold": 500, "ref_cols": ["uwwCapacity", "geometry", "POP_SERVED_EU"], "rows": 2, "one_to_one": True}
    assert captured["orchestrate"] == {
        "rows": 1,
        "approach": "2",
        "output_dir": plots_dir,
        "filename": "eu_case.gpkg",
        "pop_col": "POP_SERVED_EU",
        "save": True,
    }


def test_eu_comparison_main_returns_when_verification_dir_missing(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "verification_dir": str(tmp_path / "missing_verification"),
            "eu_plots_dir": str(tmp_path / "plots"),
            "eu_ref_filepath": str(tmp_path / "eu_ref.gpkg"),
        },
        "threshold": 500,
        "eu_reference_factor": 1,
    }

    monkeypatch.setattr(eu_comparison.os, "chdir", lambda path: None)
    monkeypatch.setattr(eu_comparison, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(eu_comparison, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(eu_comparison.os.path, "isdir", lambda path: False)
    monkeypatch.setattr(
        eu_comparison.gpd,
        "read_file",
        lambda path: (_ for _ in ()).throw(AssertionError("read_file should not be called when verification dir is missing")),
    )

    eu_comparison.main()


def test_eu_comparison_main_requires_reference_capacity_column(monkeypatch, tmp_path):
    ver_dir = str(tmp_path / "verification")
    os.makedirs(ver_dir, exist_ok=True)
    cfg = {
        "paths": {
            "verification_dir": ver_dir,
            "eu_plots_dir": str(tmp_path / "plots"),
            "eu_ref_filepath": str(tmp_path / "eu_ref.gpkg"),
        },
        "threshold": 500,
        "eu_reference_factor": 1,
    }

    ref_gdf = gpd.GeoDataFrame(
        {"other_col": [1.0], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(eu_comparison.os, "chdir", lambda path: None)
    monkeypatch.setattr(eu_comparison, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(eu_comparison, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(eu_comparison.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(eu_comparison.os, "listdir", lambda path: [])
    monkeypatch.setattr(eu_comparison.gpd, "read_file", lambda path: ref_gdf.copy())

    with pytest.raises(KeyError, match="uwwCapacity"):
        eu_comparison.main()


def test_hw_orchestrate_single_skips_2014_and_filters_quality(monkeypatch, tmp_path):
    output_dir = str(tmp_path / "plots")
    captured = {"calls": []}

    gdf = gpd.GeoDataFrame(
        {
            "POP_SERVED": [100.0, 100.0],
            "QUAL_POP": ["1.0", "0.0"],
            "2014_zonal_sum": [120.0, 120.0],
            "2015_zonal_sum": [110.0, 80.0],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(hw_comparison.os.path, "exists", lambda path: False)
    monkeypatch.setattr(hw_comparison.os, "makedirs", lambda path, exist_ok=False: captured.setdefault("makedirs", (path, exist_ok)))
    monkeypatch.setattr(
        hw_comparison,
        "composite_histogram",
        lambda data, my_dict, title, **kwargs: captured["calls"].append(
            {
                "dict": my_dict,
                "title": title,
                "cols": sorted([c for c in data.columns if c.endswith("_NDI") or c.endswith("_HW_comp")]),
                "ndi_na": int(data["2015_NDI"].isna().sum()),
                "hw_na": int(data["2015_HW_comp"].isna().sum()),
            }
        ),
    )

    hw_comparison.orchestrate_single(
        gdf,
        approach="1",
        plot_args={"save": True, "dpi": 72, "bins": 10, "fontsize": 10, "small_fontsize": 8, "lower_quantile": 0.01},
        output_dir=output_dir,
        filename="unver_case.gpkg",
        pop_col="POP_SERVED",
    )

    assert captured["makedirs"] == (output_dir, True)
    assert len(captured["calls"]) == 2
    assert list(captured["calls"][0]["dict"].keys()) == [2015]
    assert "ver: False" in captured["calls"][0]["title"]
    assert captured["calls"][0]["cols"] == ["2015_HW_comp", "2015_NDI"]
    assert captured["calls"][0]["ndi_na"] == 1
    assert captured["calls"][0]["hw_na"] == 1


def test_hw_orchestrate_single_fails_when_qual_pop_is_missing(tmp_path):
    """R8: a missing QUAL_POP used to be backfilled with '1.0' - every plant 'reported'."""
    gdf = gpd.GeoDataFrame(
        {"POP_SERVED": [100.0], "2018_zonal_sum": [110.0], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    with pytest.raises(KeyError, match="QUAL_POP"):
        hw_comparison.orchestrate_single(
            gdf, approach="1", plot_args={}, output_dir=str(tmp_path), filename="x.gpkg", pop_col="POP_SERVED",
        )


def test_hw_orchestrate_single_backfills_qual_pop_only_when_asked(monkeypatch, tmp_path):
    output_dir = str(tmp_path / "plots")
    captured = []

    gdf = gpd.GeoDataFrame(
        {
            "POP_SERVED": [100.0, 50.0],
            "2018_zonal_sum": [110.0, 40.0],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(hw_comparison.os.path, "exists", lambda path: False)
    monkeypatch.setattr(hw_comparison.os, "makedirs", lambda path, exist_ok=False: None)
    monkeypatch.setattr(
        hw_comparison,
        "composite_histogram",
        lambda data, my_dict, title, **kwargs: captured.append(data.copy()),
    )

    hw_comparison.orchestrate_single(
        gdf,
        approach="1",
        plot_args={"save": True, "dpi": 72, "bins": 10, "fontsize": 10, "small_fontsize": 8, "lower_quantile": 0.01},
        output_dir=output_dir,
        filename="single_case.gpkg",
        pop_col="POP_SERVED",
        qual_pop_default="1.0",
    )

    assert len(captured) == 2
    assert "2018_NDI" in captured[0].columns
    assert "2018_HW_comp" in captured[0].columns


def test_hw_orchestrate_single_requires_population_column(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "2018_zonal_sum": [110.0],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    with pytest.raises(KeyError, match="population column"):
        hw_comparison.orchestrate_single(
            gdf,
            approach="1",
            plot_args={"save": False, "dpi": 72, "bins": 10, "fontsize": 10, "small_fontsize": 8, "lower_quantile": 0.01},
            output_dir=str(tmp_path / "plots"),
            filename="single_case.gpkg",
            pop_col="POP_SERVED",
        )


def test_eu_orchestrate_single_builds_year_metrics_and_verified_title(monkeypatch, tmp_path):
    output_dir = str(tmp_path / "plots")
    captured = []

    gdf = gpd.GeoDataFrame(
        {
            "POP_SERVED_EU": [100.0, 50.0],
            "2014_zonal_sum": [95.0, 55.0],
            "2018_zonal_sum": [110.0, 40.0],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(hw_comparison.os.path, "exists", lambda path: False)
    monkeypatch.setattr(hw_comparison.os, "makedirs", lambda path, exist_ok=False: None)
    monkeypatch.setattr(
        hw_comparison,
        "composite_histogram",
        lambda data, my_dict, title, **kwargs: captured.append(
            {
                "dict": my_dict,
                "title": title,
                "cols": sorted([c for c in data.columns if c.endswith("_NDI") or c.endswith("_HW_comp")]),
            }
        ),
    )

    eu_comparison.orchestrate_single(
        gdf,
        approach="2",
        plot_args={"save": True, "dpi": 72, "bins": 10, "fontsize": 10, "small_fontsize": 8, "lower_quantile": 0.01},
        output_dir=output_dir,
        filename="ver_case.gpkg",
        pop_col="POP_SERVED_EU",
        qual_pop_default=None,
        filter_qual_pop=False,
        upper_quantile_hw_comp=0.95,
        comp_output_prefix="eu_comp",
        reference_name="Reference EU",
        hide_empty_axis=False,
    )

    assert len(captured) == 2
    assert list(captured[0]["dict"].keys()) == [2018]
    assert "ver: True" in captured[0]["title"]
    assert captured[0]["cols"] == ["2018_HW_comp", "2018_NDI"]


def test_eu_orchestrate_single_requires_population_column(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "2018_zonal_sum": [110.0],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    with pytest.raises(KeyError, match="population column"):
        eu_comparison.orchestrate_single(
            gdf,
            approach="2",
            plot_args={"save": False, "dpi": 72, "bins": 10, "fontsize": 10, "small_fontsize": 8, "lower_quantile": 0.01},
            output_dir=str(tmp_path / "plots"),
            filename="ver_case.gpkg",
            pop_col="POP_SERVED_EU",
        )


def test_hw_composite_histogram_save_path_calls_output_helpers(monkeypatch, tmp_path):
    output_path = str(tmp_path / "plots" / "out.png")
    captured = {}

    df = pd.DataFrame({"2020_NDI": [0.1, 0.2, 0.3]})
    my_dict = {2020: "2020_NDI"}

    monkeypatch.setattr(hw_comparison, "ensure_output_dir_for_file", lambda p: captured.setdefault("ensured", p))
    monkeypatch.setattr(hw_comparison.plt, "savefig", lambda p, dpi=None: captured.setdefault("saved", (p, dpi)))
    monkeypatch.setattr(hw_comparison.plt, "show", lambda: None)

    hw_comparison.composite_histogram(
        df,
        my_dict,
        title="t",
        output_filepath=output_path,
        save=True,
        dpi=123,
        bins=5,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    assert captured["ensured"] == output_path
    assert captured["saved"] == (output_path, 123)


def test_eu_composite_histogram_handles_missing_column_without_crashing(monkeypatch):
    df = pd.DataFrame({"other_col": [1, 2]})
    my_dict = {2021: "missing_metric"}

    class _Ax:
        def set_title(self, *_args, **_kwargs):
            return None

        def set_yticklabels(self, *_args, **_kwargs):
            return None

        def set_ylabel(self, *_args, **_kwargs):
            return None

        def set_xlabel(self, *_args, **_kwargs):
            return None

        def grid(self, *_args, **_kwargs):
            return None

        def legend(self, *_args, **_kwargs):
            return None

    class _Fig:
        def suptitle(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        hw_comparison.plt,
        "subplots",
        lambda *args, **kwargs: (_Fig(), np.array([[_Ax() for _ in range(5)] for _ in range(2)])),
    )
    monkeypatch.setattr(hw_comparison.plt, "tight_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(hw_comparison.plt, "show", lambda: None)
    monkeypatch.setattr(hw_comparison.plt, "close", lambda *args, **kwargs: None)

    # hide_empty_axis=False reproduces eu_comparison's call-site behavior;
    # the _Ax stub above has no axis() method, matching that expectation.
    hw_comparison.composite_histogram(
        df,
        my_dict,
        title="eu",
        save=False,
        bins=5,
        lower_quantile=0.0,
        upper_quantile=1.0,
        hide_empty_axis=False,
    )


def test_composite_histogram_uses_expected_rect_for_tight_layout(monkeypatch):
    df = pd.DataFrame({"2021_NDI": [0.1, 0.2, 0.3]})
    my_dict = {2021: "2021_NDI"}
    seen = {}

    class _Ax:
        def hist(self, *_args, **_kwargs):
            return None

        def axvline(self, *_args, **_kwargs):
            return None

        def set_title(self, *_args, **_kwargs):
            return None

        def set_yticklabels(self, *_args, **_kwargs):
            return None

        def set_ylabel(self, *_args, **_kwargs):
            return None

        def set_xlabel(self, *_args, **_kwargs):
            return None

        def grid(self, *_args, **_kwargs):
            return None

        def legend(self, *_args, **_kwargs):
            return None

    class _Fig:
        def suptitle(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        hw_comparison.plt,
        "subplots",
        lambda *args, **kwargs: (_Fig(), np.array([[_Ax() for _ in range(5)] for _ in range(2)])),
    )
    monkeypatch.setattr(hw_comparison.plt, "tight_layout", lambda rect=None, *args, **kwargs: seen.setdefault("rect", rect))
    monkeypatch.setattr(hw_comparison.plt, "show", lambda: None)
    monkeypatch.setattr(hw_comparison.plt, "close", lambda *args, **kwargs: None)

    hw_comparison.composite_histogram(
        df,
        my_dict,
        title="eu",
        save=False,
        bins=5,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    assert seen["rect"] == [0, 0, 1, 0.95]
