from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from wwtp.pop_at_risk_river_calculations import find_pop_in_danger_pop


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _pop_at_risk_cfg_defaults(request):
    if "mock_cfg" not in request.fixturenames:
        return

    cfg = request.getfixturevalue("mock_cfg")
    cfg.setdefault("annotations", {"max_workers": 8})


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _ImmediateExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return _ImmediateFuture(fn(*args, **kwargs))


def test_assign_tile_to_df_explodes_rows_and_clips_to_tile_bounds():
    gdf = gpd.GeoDataFrame(
        {"geometry": [box(-10.0, -10.0, 10.0, 10.0)]},
        geometry="geometry",
        crs=4326,
    )

    result = find_pop_in_danger_pop.assign_tile_to_df(gdf, zoom_level=1, max_workers=1)

    expected_tiles = find_pop_in_danger_pop.finding_tiles(gdf.geometry.iloc[0], 1)
    assert sorted(result["tile"].tolist()) == sorted(expected_tiles)
    for _, row in result.iterrows():
        tile_bbox = find_pop_in_danger_pop.find_bbox(row["tile"])
        assert row["geometry"].difference(tile_bbox).area == pytest.approx(0.0)


def test_assign_tile_to_df_rejects_non_positive_workers():
    gdf = gpd.GeoDataFrame(
        {"geometry": [box(-1.0, -1.0, 1.0, 1.0)]},
        geometry="geometry",
        crs=4326,
    )

    with pytest.raises(ValueError, match="max_workers"):
        find_pop_in_danger_pop.assign_tile_to_df(gdf, zoom_level=1, max_workers=0)


def test_group_tile_population_sums_aggregates_all_zonal_sum_columns():
    df = pd.DataFrame(
        {
            "tile": ["0-0-1", "0-0-1", "1-0-1"],
            "2020_zonal_sum": [10.0, 5.0, 7.0],
            "2021_zonal_sum": [2.0, 3.0, 4.0],
            "ignore_me": [100, 200, 300],
        }
    )

    result = find_pop_in_danger_pop.group_tile_population_sums(df).sort_values("tile").reset_index(drop=True)

    assert result.columns.tolist() == ["tile", "2020_zonal_sum", "2021_zonal_sum"]
    assert result.loc[0, "2020_zonal_sum"] == pytest.approx(15.0)
    assert result.loc[0, "2021_zonal_sum"] == pytest.approx(5.0)
    assert result.loc[1, "2020_zonal_sum"] == pytest.approx(7.0)
    assert result.loc[1, "2021_zonal_sum"] == pytest.approx(4.0)


def test_rename_cols_prefixes_non_geometry_columns():
    gdf = gpd.GeoDataFrame(
        {
            "tile": ["0-0-1"],
            "geometry": [box(0, 0, 1, 1)],
            "2020_zonal_sum": [12.0],
            "country": ["DE"],
        },
        geometry="geometry",
        crs=4326,
    )

    result = find_pop_in_danger_pop.rename_cols(gdf, "1000")

    assert "tile" in result.columns
    assert "geometry" in result.columns
    assert "1000_2020_zonal_sum" in result.columns
    assert "1000_country" in result.columns


def test_find_tiles_in_countries_collects_parallel_country_results(monkeypatch, tiny_country_gdf):
    monkeypatch.setattr(find_pop_in_danger_pop, "ThreadPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(find_pop_in_danger_pop, "as_completed", lambda futures: futures)

    def fake_find_tiles_in_a_country(country_polygon, country, country_id_col, zoom_level):
        return gpd.GeoDataFrame(
            {
                "tile": [f"{country}-seed-{zoom_level}"],
                country_id_col: [country],
                "geometry": [country_polygon],
            },
            geometry="geometry",
            crs=4326,
        )

    monkeypatch.setattr(find_pop_in_danger_pop, "find_tiles_in_a_country", fake_find_tiles_in_a_country)

    result = find_pop_in_danger_pop.find_tiles_in_countries(
        tiny_country_gdf,
        zoom_level=4,
        country_id_col="country",
        max_workers=2,
    )

    assert set(result["country"]) == {"DE", "FR"}
    assert set(result["tile"]) == {"DE-seed-4", "FR-seed-4"}


def test_find_tiles_in_countries_rejects_non_positive_workers(tiny_country_gdf):
    with pytest.raises(ValueError, match="max_workers"):
        find_pop_in_danger_pop.find_tiles_in_countries(
            tiny_country_gdf,
            zoom_level=4,
            country_id_col="country",
            max_workers=0,
        )


def test_group_tile_population_sums_returns_input_when_tile_column_missing():
    df = pd.DataFrame({"value": [1, 2], "2020_zonal_sum": [3.0, 4.0]})

    result = find_pop_in_danger_pop.group_tile_population_sums(df)

    assert result.equals(df)


def test_group_tile_population_sums_returns_input_when_no_zonal_sum_columns():
    df = pd.DataFrame({"tile": ["0-0-1"], "value": [1]})

    result = find_pop_in_danger_pop.group_tile_population_sums(df)

    assert result.equals(df)


def test_group_tile_population_sums_keeps_all_nan_groups_as_nan():
    df = pd.DataFrame(
        {
            "tile": ["0-0-1", "0-0-1", "1-0-1"],
            "2020_zonal_sum": [np.nan, np.nan, 3.0],
        }
    )

    result = find_pop_in_danger_pop.group_tile_population_sums(df).sort_values("tile").reset_index(drop=True)

    assert pd.isna(result.loc[0, "2020_zonal_sum"])
    assert result.loc[1, "2020_zonal_sum"] == pytest.approx(3.0)


def test_main_writes_output_when_final_merge_has_cross_radius_nan(monkeypatch, mock_cfg, tmp_path):
    cfg = mock_cfg
    cfg["zoom_level"] = 4
    cfg["paths"]["impact_pop_polygons_outpath"] = str(tmp_path / "impact_polygons.gpkg")
    cfg["paths"]["pop_at_risk_output_filepath"] = str(tmp_path / "pop_at_risk.parquet")
    cfg["paths"]["pop_tif_dir"] = str(tmp_path / "tifs")
    cfg["paths"]["overture"] = str(tmp_path / "overture.parquet")
    cfg["country_boundary_column"] = "country"
    cfg["country_output_column"] = "ISO_2"
    rng = np.random.default_rng(53)
    saved = {}
    grouped_results = iter(
        [
            pd.DataFrame({"tile": ["0-0-1"], "2020_zonal_sum": [float(rng.integers(1, 20))]}),
            pd.DataFrame({"tile": ["1-0-1"], "2020_zonal_sum": [float(rng.integers(20, 40))]}),
        ]
    )

    input_gdf = gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs=4326,
    )

    monkeypatch.setattr(find_pop_in_danger_pop.os, "chdir", lambda path: None)
    monkeypatch.setattr(find_pop_in_danger_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_pop_in_danger_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(
        find_pop_in_danger_pop,
        "glob",
        lambda pattern: [
            os.path.join(tmp_path, "impact_polygons_1000.gpkg"),
            os.path.join(tmp_path, "impact_polygons_2000.gpkg"),
        ],
    )
    monkeypatch.setattr(find_pop_in_danger_pop.gpd, "read_file", lambda path: input_gdf.copy())
    monkeypatch.setattr(find_pop_in_danger_pop, "assign_tile_to_df", lambda gdf, zoom_level, max_workers: gdf.copy())
    monkeypatch.setattr(
        find_pop_in_danger_pop, "intersects_with_country_db", lambda gdf, *args, **kwargs: gdf.assign(ISO_2="DE")
    )
    monkeypatch.setattr(find_pop_in_danger_pop, "intersect_all_files", lambda gdf, *args, **kwargs: gdf.copy())
    monkeypatch.setattr(find_pop_in_danger_pop, "group_tile_population_sums", lambda gdf: next(grouped_results))
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: None)

    original_to_file = gpd.GeoDataFrame.to_file
    original_to_parquet = gpd.GeoDataFrame.to_parquet

    def fake_to_file(self, path, index=False, driver=None, **kwargs):
        saved.setdefault("intermediate_writes", []).append({"path": path, "rows": len(self)})

    def fake_to_parquet(self, path, engine=None, index=False, **kwargs):
        saved["final"] = {"path": path, "engine": engine, "index": index, "frame": self.copy()}

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fake_to_parquet)
        find_pop_in_danger_pop.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", original_to_parquet)

    assert saved["final"]["path"] == cfg["paths"]["pop_at_risk_output_filepath"]


def test_main_writes_empty_output_when_no_input_files_match(monkeypatch, mock_cfg, tmp_path):
    cfg = mock_cfg
    cfg["zoom_level"] = 4
    cfg["paths"]["impact_pop_polygons_outpath"] = str(tmp_path / "impact_polygons.gpkg")
    cfg["paths"]["pop_at_risk_output_filepath"] = str(tmp_path / "pop_at_risk.parquet")
    cfg["paths"]["pop_tif_dir"] = str(tmp_path / "tifs")
    cfg["paths"]["overture"] = str(tmp_path / "overture.parquet")
    cfg["country_boundary_column"] = "country"
    cfg["country_output_column"] = "ISO_2"
    saved = {}

    monkeypatch.setattr(find_pop_in_danger_pop.os, "chdir", lambda path: None)
    monkeypatch.setattr(find_pop_in_danger_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_pop_in_danger_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(find_pop_in_danger_pop, "glob", lambda pattern: [])
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: saved.setdefault("ensured", path))

    original_to_parquet = gpd.GeoDataFrame.to_parquet

    def fake_to_parquet(self, path, engine=None, index=False, **kwargs):
        saved["final"] = {"path": path, "engine": engine, "index": index, "frame": self.copy()}

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fake_to_parquet)
        find_pop_in_danger_pop.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", original_to_parquet)

    assert saved["ensured"] == cfg["paths"]["pop_at_risk_output_filepath"]
    assert saved["final"]["path"] == cfg["paths"]["pop_at_risk_output_filepath"]
    assert saved["final"]["engine"] == "pyarrow"
    assert saved["final"]["index"] is False
    assert saved["final"]["frame"].empty
    assert "geometry" in saved["final"]["frame"].columns


def test_main_rejects_missing_annotations_max_workers(monkeypatch, mock_cfg, tmp_path):
    cfg = mock_cfg
    cfg.pop("annotations", None)
    cfg["zoom_level"] = 4
    cfg["paths"]["impact_pop_polygons_outpath"] = str(tmp_path / "impact_polygons.gpkg")
    cfg["paths"]["pop_at_risk_output_filepath"] = str(tmp_path / "pop_at_risk.parquet")
    cfg["paths"]["pop_tif_dir"] = str(tmp_path / "tifs")
    cfg["paths"]["overture"] = str(tmp_path / "overture.parquet")
    cfg["country_boundary_column"] = "country"
    cfg["country_output_column"] = "ISO_2"
    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, geometry="geometry", crs=4326)

    monkeypatch.setattr(find_pop_in_danger_pop.os, "chdir", lambda path: None)
    monkeypatch.setattr(find_pop_in_danger_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_pop_in_danger_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(find_pop_in_danger_pop, "glob", lambda pattern: [str(tmp_path / "impact_polygons_1000.gpkg")])
    monkeypatch.setattr(find_pop_in_danger_pop.gpd, "read_file", lambda path: gdf.copy())

    monkeypatch.setattr(
        find_pop_in_danger_pop, "assign_tile_to_df", lambda frame, zoom_level, max_workers: frame.copy()
    )
    monkeypatch.setattr(
        find_pop_in_danger_pop, "intersects_with_country_db", lambda frame, *args, **kwargs: frame.assign(ISO_2="DE")
    )
    monkeypatch.setattr(
        find_pop_in_danger_pop,
        "intersect_all_files",
        lambda frame, tif_dir, max_workers, all_years=False, country_col=None: frame.copy(),
    )
    monkeypatch.setattr(
        find_pop_in_danger_pop,
        "group_tile_population_sums",
        lambda frame: pd.DataFrame({"tile": ["0-0-1"], "2020_zonal_sum": [1.0]}),
    )
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: None)

    original_to_file = gpd.GeoDataFrame.to_file
    original_to_parquet = gpd.GeoDataFrame.to_parquet
    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, *args, **kwargs: None)
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", lambda self, *args, **kwargs: None)
        with pytest.raises(KeyError, match="annotations"):
            find_pop_in_danger_pop.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", original_to_parquet)


def test_main_uses_configured_annotations_max_workers(monkeypatch, mock_cfg, tmp_path):
    cfg = mock_cfg
    cfg["zoom_level"] = 4
    cfg["annotations"] = {"max_workers": 3}
    cfg["paths"]["impact_pop_polygons_outpath"] = str(tmp_path / "impact_polygons.gpkg")
    cfg["paths"]["pop_at_risk_output_filepath"] = str(tmp_path / "pop_at_risk.parquet")
    cfg["paths"]["pop_tif_dir"] = str(tmp_path / "tifs")
    cfg["paths"]["overture"] = str(tmp_path / "overture.parquet")
    cfg["country_boundary_column"] = "country"
    cfg["country_output_column"] = "ISO_2"

    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, geometry="geometry", crs=4326)
    seen = {}

    monkeypatch.setattr(find_pop_in_danger_pop.os, "chdir", lambda path: None)
    monkeypatch.setattr(find_pop_in_danger_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_pop_in_danger_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(find_pop_in_danger_pop, "glob", lambda pattern: [str(tmp_path / "impact_polygons_1000.gpkg")])
    monkeypatch.setattr(find_pop_in_danger_pop.gpd, "read_file", lambda path: gdf.copy())

    def _assign_tile(frame, zoom_level, max_workers):
        seen["max_workers"] = max_workers
        return frame.copy()

    monkeypatch.setattr(find_pop_in_danger_pop, "assign_tile_to_df", _assign_tile)
    monkeypatch.setattr(
        find_pop_in_danger_pop, "intersects_with_country_db", lambda frame, *args, **kwargs: frame.assign(ISO_2="DE")
    )
    monkeypatch.setattr(find_pop_in_danger_pop, "intersect_all_files", lambda frame, *args, **kwargs: frame.copy())
    monkeypatch.setattr(
        find_pop_in_danger_pop,
        "group_tile_population_sums",
        lambda frame: pd.DataFrame({"tile": ["0-0-1"], "2020_zonal_sum": [1.0]}),
    )
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: None)

    original_to_file = gpd.GeoDataFrame.to_file
    original_to_parquet = gpd.GeoDataFrame.to_parquet
    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, *args, **kwargs: None)
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", lambda self, *args, **kwargs: None)
        find_pop_in_danger_pop.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", original_to_parquet)

    assert seen["max_workers"] == 3


def test_main_uses_at_least_one_worker_for_intersections(monkeypatch, mock_cfg, tmp_path):
    cfg = mock_cfg
    cfg["zoom_level"] = 4
    cfg["annotations"] = {"max_workers": 1}
    cfg["paths"]["impact_pop_polygons_outpath"] = str(tmp_path / "impact_polygons.gpkg")
    cfg["paths"]["pop_at_risk_output_filepath"] = str(tmp_path / "pop_at_risk.parquet")
    cfg["paths"]["pop_tif_dir"] = str(tmp_path / "tifs")
    cfg["paths"]["overture"] = str(tmp_path / "overture.parquet")
    cfg["country_boundary_column"] = "country"
    cfg["country_output_column"] = "ISO_2"

    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, geometry="geometry", crs=4326)
    seen = {}

    monkeypatch.setattr(find_pop_in_danger_pop.os, "chdir", lambda path: None)
    monkeypatch.setattr(find_pop_in_danger_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_pop_in_danger_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(find_pop_in_danger_pop, "glob", lambda pattern: [str(tmp_path / "impact_polygons_1000.gpkg")])
    monkeypatch.setattr(find_pop_in_danger_pop.gpd, "read_file", lambda path: gdf.copy())
    monkeypatch.setattr(
        find_pop_in_danger_pop, "assign_tile_to_df", lambda frame, zoom_level, max_workers: frame.copy()
    )
    monkeypatch.setattr(
        find_pop_in_danger_pop, "intersects_with_country_db", lambda frame, *args, **kwargs: frame.assign(ISO_2="DE")
    )

    def _intersect(frame, tif_dir, max_workers, all_years=False, country_col=None):
        seen["intersect_workers"] = max_workers
        return frame.copy()

    monkeypatch.setattr(find_pop_in_danger_pop, "intersect_all_files", _intersect)
    monkeypatch.setattr(
        find_pop_in_danger_pop,
        "group_tile_population_sums",
        lambda frame: pd.DataFrame({"tile": ["0-0-1"], "2020_zonal_sum": [1.0]}),
    )
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: None)

    original_to_file = gpd.GeoDataFrame.to_file
    original_to_parquet = gpd.GeoDataFrame.to_parquet
    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, *args, **kwargs: None)
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", lambda self, *args, **kwargs: None)
        find_pop_in_danger_pop.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", original_to_parquet)

    assert seen["intersect_workers"] == 1


def test_find_tiles_in_a_country_clips_tiles_and_assigns_country_id():
    country_polygon = box(-1.0, -1.0, 1.0, 1.0)

    result = find_pop_in_danger_pop.find_tiles_in_a_country(
        country_polygon,
        "DE",
        "ISO_2",
        zoom_level=1,
    )

    assert not result.empty
    assert set(result["ISO_2"]) == {"DE"}
    assert result.geometry.notna().all()


def test_find_tiles_in_countries_returns_empty_dataframe_for_empty_input():
    countries = gpd.GeoDataFrame(
        {"ISO_2": [], "geometry": []},
        geometry="geometry",
        crs=4326,
    )

    result = find_pop_in_danger_pop.find_tiles_in_countries(
        countries, zoom_level=4, country_id_col="ISO_2", max_workers=1
    )

    assert result.empty
    assert result.columns.tolist() == ["tile", "geometry", "ISO_2"]


def test_assign_tile_to_df_returns_empty_input_unchanged():
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=4326)

    result = find_pop_in_danger_pop.assign_tile_to_df(empty, zoom_level=4, max_workers=1)

    assert result.empty
    assert result.columns.tolist() == empty.columns.tolist()


def test_main_writes_empty_tile_groups_when_grouping_drops_tile_column(monkeypatch, mock_cfg, tmp_path):
    cfg = mock_cfg
    cfg["zoom_level"] = 4
    cfg["paths"]["impact_pop_polygons_outpath"] = str(tmp_path / "impact_polygons.gpkg")
    cfg["paths"]["pop_at_risk_output_filepath"] = str(tmp_path / "pop_at_risk.parquet")
    cfg["paths"]["pop_tif_dir"] = str(tmp_path / "tifs")
    cfg["paths"]["overture"] = str(tmp_path / "overture.parquet")
    cfg["country_boundary_column"] = "country"
    cfg["country_output_column"] = "ISO_2"
    saved = {}

    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, geometry="geometry", crs=4326)

    monkeypatch.setattr(find_pop_in_danger_pop.os, "chdir", lambda path: None)
    monkeypatch.setattr(find_pop_in_danger_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_pop_in_danger_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(find_pop_in_danger_pop, "glob", lambda pattern: [str(tmp_path / "impact_polygons_1000.gpkg")])
    monkeypatch.setattr(find_pop_in_danger_pop.gpd, "read_file", lambda path: gdf.copy())
    monkeypatch.setattr(
        find_pop_in_danger_pop, "assign_tile_to_df", lambda frame, zoom_level, max_workers: frame.copy()
    )
    monkeypatch.setattr(
        find_pop_in_danger_pop, "intersects_with_country_db", lambda frame, *args, **kwargs: frame.assign(ISO_2="DE")
    )
    monkeypatch.setattr(find_pop_in_danger_pop, "intersect_all_files", lambda frame, *args, **kwargs: frame.copy())
    monkeypatch.setattr(
        find_pop_in_danger_pop, "group_tile_population_sums", lambda frame: pd.DataFrame({"2020_zonal_sum": [1.0]})
    )
    monkeypatch.setattr(
        "wwtp.utils.ensure_output_dir_for_file", lambda path: saved.setdefault("ensured", []).append(path)
    )

    original_to_file = gpd.GeoDataFrame.to_file
    original_to_parquet = gpd.GeoDataFrame.to_parquet

    def fake_to_parquet(self, path, engine=None, index=False, **kwargs):
        saved["final"] = {"path": path, "engine": engine, "index": index, "frame": self.copy()}

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, *args, **kwargs: None)
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fake_to_parquet)
        find_pop_in_danger_pop.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", original_to_parquet)

    assert saved["ensured"][-1] == cfg["paths"]["pop_at_risk_output_filepath"]
    assert saved["final"]["frame"].empty
    assert saved["final"]["frame"].columns.tolist() == ["tile", "geometry"]


def test_find_pop_in_danger_script_entrypoint_runs_main_guard(monkeypatch):
    import runpy

    import wwtp.starter as starter_mod

    module_name = "wwtp.pop_at_risk_river_calculations.find_pop_in_danger_pop"

    monkeypatch.setattr(starter_mod, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(
        starter_mod,
        "load_config",
        lambda **overrides: {"zoom_level": 8, "annotations": {"max_workers": 1}, "paths": {}},
    )

    with pytest.raises(KeyError, match="pop_tif_dir"):
        runpy.run_module(module_name, run_name="__main__")
