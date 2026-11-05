from __future__ import annotations

import sys

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon, box

from src.industrial_analysis import find_unconnected_industrial_areas


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _industrial_cfg_defaults(request):
    if "mock_cfg" not in request.fixturenames:
        return

    cfg = request.getfixturevalue("mock_cfg")
    cfg.setdefault("basin_column_name", "HYBAS_ID")
    cfg.setdefault("prepare_data_fn", "prepare_data")
    cfg.setdefault("n_points", 250)
    cfg.setdefault("threshold", 500)
    cfg.setdefault("sigma", 3)
    cfg.setdefault("percent_threshold", 10)
    cfg.setdefault("old_site_id_column", "OLD_WASTE_ID")
    cfg.setdefault("calculate_area_fn", "calculate_area")
    cfg.setdefault("calculate_buffer_fn", "calculate_buffer")
    cfg.setdefault("area_fn_kwargs", {})
    cfg.setdefault("temp_voronoi_overwrite", False)
    cfg.setdefault("flush_size", 1000)
    cfg.setdefault("min_buffer", 2000)
    cfg.setdefault("max_buffer", 50000)
    cfg.setdefault("k_min", 0.32)
    cfg.setdefault("k_max", 0.92)
    cfg.setdefault("detection_confidence_threshold", 3)
    cfg.setdefault("industrial_min_cells", 100)
    cfg.setdefault("industrial_category_numbers", [2, 4])
    cfg.setdefault("mix_use_categories", [6])
    cfg.setdefault("overwrite_existing", False)
    cfg.setdefault("unconnected_sjoin_workers", 8)
    cfg.setdefault("scipy_true", False)
    cfg.setdefault("cv2_true", False)
    cfg.setdefault("csv_files", False)
    cfg.setdefault("remove_industrial", False)
    cfg.setdefault("return_boolean", True)

    paths = cfg.setdefault("paths", {})
    paths.setdefault("industrial_merged_filepath", "industrial_merged.gpkg")
    paths.setdefault("industrial_unconnected_output", "industrial_unconnected.parquet")
    paths.setdefault("annotated_all_filepath", "corrected_all.gpkg")
    paths.setdefault("watershed", "watershed.gpkg")
    paths.setdefault("bboxes", "bboxes.csv")
    paths.setdefault("hydrowaste", "hydrowaste.csv")
    paths.setdefault("buffers_dir", "buffers")
    paths.setdefault("voronoi_dir", "voronoi")
    paths.setdefault("industrial_buffers_dir", "industrial_buffers")
    paths.setdefault("industrial_voronoi_dir", "industrial_voronoi")
    paths.setdefault("overture", "overture.parquet")
    paths.setdefault("overture_s3_url", "s3://example/overture")


def test_filter_industrial_wwtps_keeps_only_matching_categories(mock_cfg, tiny_points_gdf):
    cfg = mock_cfg
    cfg["industrial_category_numbers"] = ["1", "4"]

    result = find_unconnected_industrial_areas.filter_industrial_wwtps(cfg, tiny_points_gdf)

    assert set(result["WASTE_ID"]) == {1, 6}


def test_filter_industrial_wwtps_rejects_an_empty_category_list(mock_cfg, tiny_points_gdf):
    """R10: an empty list used to mean "every plant", i.e. a copy of the main run."""
    mock_cfg["industrial_category_numbers"] = []

    with pytest.raises(ValueError, match="industrial_category_numbers is empty"):
        find_unconnected_industrial_areas.filter_industrial_wwtps(mock_cfg, tiny_points_gdf)


def test_industrial_output_paths_never_point_at_the_main_voronoi_outputs(mock_cfg):
    """R10: the industrial run used the create_voronoi file names and directories."""
    from src.pipelines import create_output_paths

    for key in ("level", "version", "buffer_path_token", "weight_type", "weight_func_suffix"):
        mock_cfg.setdefault(key, "x")
    main = create_output_paths(mock_cfg)
    industrial = find_unconnected_industrial_areas.industrial_output_paths(mock_cfg)

    for group in ("voronoi", "buffers"):
        for key, path in industrial[group].items():
            assert path != main[group][key]
            assert path.startswith("industrial_")


def test_find_unconnected_areas_returns_only_polygons_outside_service_regions():
    industrial_gdf = gpd.GeoDataFrame(
        {
            "industrial_id": [1, 2],
            "geometry": [box(0.0, 0.0, 0.05, 0.05), box(2.0, 2.0, 2.1, 2.1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    voronoi_gdf = gpd.GeoDataFrame(
        {"geometry": [box(-0.1, -0.1, 0.1, 0.1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = find_unconnected_industrial_areas.find_unconnected_areas(industrial_gdf, voronoi_gdf)

    assert result["industrial_id"].tolist() == [2]


def test_load_industrial_areas_returns_none_when_file_missing(mock_cfg, monkeypatch):
    cfg = mock_cfg
    monkeypatch.setattr(find_unconnected_industrial_areas.os.path, "exists", lambda path: False)

    result = find_unconnected_industrial_areas.load_industrial_areas(cfg)

    assert result is None


def test_load_industrial_areas_reads_geopackage_when_present(mock_cfg, monkeypatch):
    cfg = mock_cfg
    expected = gpd.GeoDataFrame(
        {"industrial_id": [1], "geometry": [box(0.0, 0.0, 1.0, 1.0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    captured = {}

    monkeypatch.setattr(find_unconnected_industrial_areas.os.path, "exists", lambda path: True)

    def fake_read_file(path, driver=None):
        captured["path"] = path
        captured["driver"] = driver
        return expected

    monkeypatch.setattr(find_unconnected_industrial_areas.gpd, "read_file", fake_read_file)

    result = find_unconnected_industrial_areas.load_industrial_areas(cfg)

    assert result.equals(expected)
    assert captured == {
        "path": cfg["paths"]["industrial_merged_filepath"],
        "driver": "GPKG",
    }


def test_load_wwtps_adds_basin_info_when_missing_for_approach_one(mock_cfg, monkeypatch):
    cfg = mock_cfg
    basin_col = cfg["basin_column_name"]
    read_calls = []

    wwtps_gdf = gpd.GeoDataFrame(
        {
            "WASTE_ID": [1, 2],
            basin_col: [None, None],
            "geometry": [Point(0.0, 0.0), Point(1.0, 1.0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    basin_gdf = gpd.GeoDataFrame(
        {
            basin_col: [101, 202],
            "geometry": [box(-1.0, -1.0, 0.5, 0.5), box(0.5, 0.5, 2.0, 2.0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    def fake_read_file(path, driver=None):
        read_calls.append((path, driver))
        if path == cfg["paths"]["annotated_all_filepath"]:
            return wwtps_gdf.copy()
        if path == cfg["paths"]["watershed"]:
            return basin_gdf.copy()
        raise AssertionError(f"Unexpected path: {path}")

    def fake_intersect(run_gdf, basin_slice, column_name, concurrency):
        assert column_name == basin_col
        assert basin_slice.columns.tolist() == [basin_col, "geometry"]
        assert concurrency == cfg["sindex_concurrency"]
        return run_gdf.assign(**{basin_col: [101, 202]})

    monkeypatch.setattr(find_unconnected_industrial_areas.gpd, "read_file", fake_read_file)
    monkeypatch.setattr(find_unconnected_industrial_areas, "intersect_with_polygon_sindex", fake_intersect)

    result = find_unconnected_industrial_areas.load_wwtps(cfg, approach_id="1")

    assert result[basin_col].tolist() == [101, 202]
    assert read_calls == [
        (cfg["paths"]["annotated_all_filepath"], "GPKG"),
        (cfg["paths"]["watershed"], "GPKG"),
    ]


def test_load_wwtps_raises_when_basin_dataset_lacks_configured_column(mock_cfg, monkeypatch):
    cfg = mock_cfg
    basin_col = cfg["basin_column_name"]

    wwtps_gdf = gpd.GeoDataFrame(
        {
            "WASTE_ID": [1],
            basin_col: [None],
            "geometry": [Point(0.0, 0.0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    basin_gdf = gpd.GeoDataFrame(
        {"wrong_column": [1], "geometry": [box(-1.0, -1.0, 1.0, 1.0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    def fake_read_file(path, driver=None):
        if path == cfg["paths"]["annotated_all_filepath"]:
            return wwtps_gdf.copy()
        if path == cfg["paths"]["watershed"]:
            return basin_gdf.copy()
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(find_unconnected_industrial_areas.gpd, "read_file", fake_read_file)

    with pytest.raises(KeyError, match="Configured basin column"):
        find_unconnected_industrial_areas.load_wwtps(cfg, approach_id="1")


def test_run_voronoi_for_wwtps_approach_one_builds_basin_buffer_ids(
    mock_cfg,
    tiny_points_gdf,
    tiny_watershed_gdf,
    tiny_country_gdf,
    monkeypatch,
):
    cfg = mock_cfg
    cfg["weight_func"] = "mult"
    cfg["weight_method"] = "logarithmic"
    cfg["distance_fn"] = "distance-sentinel"
    captured = {}

    def fake_run_voronoi_approach(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        region_df = gpd.GeoDataFrame(
            {"geometry": [box(-0.08, -0.08, 0.08, 0.08)]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        return region_df, None

    monkeypatch.setattr(find_unconnected_industrial_areas, "run_voronoi_approach", fake_run_voronoi_approach)

    result = find_unconnected_industrial_areas.run_voronoi_for_wwtps(
        cfg,
        "1",
        tiny_points_gdf.iloc[:3].copy(),
        tiny_watershed_gdf.copy(),
        tiny_country_gdf.copy(),
        {"buffers": {"WWTP": "unused"}},
        output_path="synthetic.gpkg",
        only_round=True,
    )

    run_gdf = captured["args"][1]
    clipping_gdf = captured["args"][2]

    assert result is not None
    assert run_gdf["buffer_id"].tolist() == run_gdf["HYBAS_ID"].tolist()
    assert clipping_gdf["buffer_id"].tolist() == clipping_gdf["HYBAS_ID"].tolist()
    assert captured["kwargs"]["scale_weights"] is True
    assert captured["kwargs"]["buffering"] is True
    assert captured["kwargs"]["only_round"] is True
    assert captured["kwargs"]["method"] == "logarithmic"


def test_run_voronoi_for_wwtps_approach_zero_uses_dissolved_buffers(
    mock_cfg,
    tiny_points_gdf,
    tiny_country_gdf,
    monkeypatch,
):
    cfg = mock_cfg
    cfg["weight_func"] = ""
    cfg["weight_method"] = "linear"
    cfg["distance_fn"] = "distance-sentinel"
    captured = {}

    dissolved_buffers = gpd.GeoDataFrame(
        {
            cfg["country_output_column"]: ["DE", "FR"],
            "geometry": [box(-0.1, -0.1, 0.1, 0.1), box(0.9, -0.1, 1.1, 0.1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(
        find_unconnected_industrial_areas,
        "orchestrate_overlaps",
        lambda gdf, max_workers, output_path, buffer, country_col: dissolved_buffers.copy(),
    )
    monkeypatch.setattr(find_unconnected_industrial_areas, "drop_duplicates", lambda gdf, col: gdf)
    monkeypatch.setattr(
        find_unconnected_industrial_areas,
        "intersect_with_polygon_sindex",
        lambda run_gdf, buffers, col, concurrency: run_gdf.assign(buffer_id=[0] * len(run_gdf)),
    )

    def fake_run_voronoi_approach(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        region_df = gpd.GeoDataFrame(
            {"geometry": [box(-0.1, -0.1, 0.1, 0.1)]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        return region_df, None

    monkeypatch.setattr(find_unconnected_industrial_areas, "run_voronoi_approach", fake_run_voronoi_approach)

    result = find_unconnected_industrial_areas.run_voronoi_for_wwtps(
        cfg,
        "0",
        tiny_points_gdf.iloc[:2].copy(),
        gpd.GeoDataFrame({"HYBAS_ID": [101], "geometry": [box(-1, -1, 1, 1)]}, geometry="geometry", crs="EPSG:4326"),
        tiny_country_gdf.copy(),
        {"buffers": {"WWTP": "buffers.gpkg"}},
        output_path="synthetic.gpkg",
        only_round=False,
    )

    run_gdf = captured["args"][1]
    clipping_gdf = captured["args"][2]

    assert result is not None
    assert run_gdf["buffer_id"].tolist() == [0, 0]
    assert clipping_gdf["buffer_id"].tolist() == [0, 1]
    assert captured["kwargs"]["scale_weights"] is False
    assert captured["kwargs"]["buffering"] is False
    assert captured["kwargs"]["method"] == "linear"


def test_run_voronoi_for_wwtps_rejects_invalid_approach(mock_cfg, tiny_points_gdf, tiny_watershed_gdf, tiny_country_gdf):
    cfg = mock_cfg

    with pytest.raises(ValueError, match="Unsupported approach"):
        find_unconnected_industrial_areas.run_voronoi_for_wwtps(
            cfg,
            "2",
            tiny_points_gdf.copy(),
            tiny_watershed_gdf.copy(),
            tiny_country_gdf.copy(),
            {"buffers": {"WWTP": "unused"}},
            output_path="synthetic.gpkg",
            only_round=False,
        )


def test_main_skips_when_output_exists_and_overwrite_disabled(mock_cfg, monkeypatch, tmp_path):
    cfg = mock_cfg
    cfg["paths"]["industrial_unconnected_output"] = str(tmp_path / "industrial.parquet")
    cfg["overwrite_existing"] = False

    monkeypatch.setattr(sys, "argv", ["find_unconnected_industrial_areas.py"])
    monkeypatch.setattr(find_unconnected_industrial_areas, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(find_unconnected_industrial_areas, "industrial_output_paths", lambda config: {"voronoi": {"1": "unused.gpkg"}})
    monkeypatch.setattr(
        find_unconnected_industrial_areas.os.path,
        "exists",
        lambda path: path == cfg["paths"]["industrial_unconnected_output"],
    )
    monkeypatch.setattr(
        find_unconnected_industrial_areas,
        "load_industrial_areas",
        lambda config: (_ for _ in ()).throw(AssertionError("load_industrial_areas should not be called")),
    )

    assert find_unconnected_industrial_areas.main() is True


def test_main_writes_industrial_copy_when_no_industrial_wwtps(mock_cfg, monkeypatch, tmp_path, tiny_points_gdf):
    cfg = mock_cfg
    cfg["paths"]["industrial_unconnected_output"] = str(tmp_path / "industrial.parquet")
    cfg["overwrite_existing"] = True

    industrial_gdf = gpd.GeoDataFrame(
        {
            "industrial_id": [1, 2],
            "geometry": [box(0.0, 0.0, 0.1, 0.1), box(1.0, 1.0, 1.1, 1.1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    saved = {}

    monkeypatch.setattr(sys, "argv", ["find_unconnected_industrial_areas.py"])
    monkeypatch.setattr(find_unconnected_industrial_areas, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(find_unconnected_industrial_areas, "industrial_output_paths", lambda config: {"voronoi": {"1": "unused.gpkg"}})
    monkeypatch.setattr(find_unconnected_industrial_areas.os.path, "exists", lambda path: False)
    monkeypatch.setattr(find_unconnected_industrial_areas.os, "makedirs", lambda path, exist_ok=False: None)
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_industrial_areas", lambda config: industrial_gdf.copy())
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_wwtps", lambda config, approach_id: tiny_points_gdf.copy())
    monkeypatch.setattr(
        find_unconnected_industrial_areas,
        "filter_industrial_wwtps",
        lambda config, wwtps_gdf: wwtps_gdf.iloc[0:0].copy(),
    )

    def fake_to_parquet(self, path, index=False, **kwargs):
        saved["path"] = path
        saved["index"] = index
        saved["industrial_ids"] = self["industrial_id"].tolist()

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fake_to_parquet)

    assert find_unconnected_industrial_areas.main() is True
    assert saved == {
        "path": cfg["paths"]["industrial_unconnected_output"],
        "index": False,
        "industrial_ids": [1, 2],
    }


def test_main_returns_false_when_voronoi_generation_fails(
    mock_cfg,
    monkeypatch,
    tmp_path,
    tiny_points_gdf,
    tiny_watershed_gdf,
    tiny_country_gdf,
):
    cfg = mock_cfg
    cfg["paths"]["industrial_unconnected_output"] = str(tmp_path / "industrial.parquet")
    cfg["overwrite_existing"] = True
    cfg["prepare_data_fn"] = "prepare_data"

    industrial_gdf = gpd.GeoDataFrame(
        {"industrial_id": [1], "geometry": [box(0.0, 0.0, 0.1, 0.1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(sys, "argv", ["find_unconnected_industrial_areas.py"])
    monkeypatch.setattr(find_unconnected_industrial_areas, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(find_unconnected_industrial_areas, "industrial_output_paths", lambda config: {"voronoi": {"1": "voronoi.gpkg"}})
    monkeypatch.setattr(find_unconnected_industrial_areas.os.path, "exists", lambda path: False)
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_industrial_areas", lambda config: industrial_gdf.copy())
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_wwtps", lambda config, approach_id: tiny_points_gdf.copy())
    monkeypatch.setattr(find_unconnected_industrial_areas, "filter_industrial_wwtps", lambda config, wwtps_gdf: wwtps_gdf.copy())
    monkeypatch.setattr(
        find_unconnected_industrial_areas,
        "_resolve_configured_callable",
        lambda *args, **kwargs: (lambda config: {"country_df": tiny_country_gdf.copy(), "basin_gdf": tiny_watershed_gdf.copy()}),
    )
    monkeypatch.setattr(find_unconnected_industrial_areas, "run_voronoi_for_wwtps", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gpd.GeoDataFrame,
        "to_parquet",
        lambda self, path, index=False, **kwargs: (_ for _ in ()).throw(AssertionError("to_parquet should not be called")),
    )

    assert find_unconnected_industrial_areas.main() is False


def test_run_voronoi_for_wwtps_requires_basin_column_in_wwtp_df(
    mock_cfg,
    tiny_watershed_gdf,
    tiny_country_gdf,
):
    cfg = mock_cfg
    missing_basin = gpd.GeoDataFrame(
        {
            "WASTE_ID": [1],
            "geometry": [Point(0.0, 0.0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    with pytest.raises(KeyError, match="WWTP dataframe"):
        find_unconnected_industrial_areas.run_voronoi_for_wwtps(
            cfg,
            "1",
            missing_basin,
            tiny_watershed_gdf.copy(),
            tiny_country_gdf.copy(),
            {"buffers": {"WWTP": "unused"}},
            output_path="synthetic.gpkg",
            only_round=False,
        )


def test_find_unconnected_areas_reprojects_voronoi_when_crs_differs():
    industrial_gdf = gpd.GeoDataFrame(
        {
            "industrial_id": [1, 2],
            "geometry": [box(0.0, 0.0, 1000.0, 1000.0), box(200000.0, 200000.0, 201000.0, 201000.0)],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    voronoi_gdf = gpd.GeoDataFrame(
        {"geometry": [box(-0.1, -0.1, 0.1, 0.1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = find_unconnected_industrial_areas.find_unconnected_areas(industrial_gdf, voronoi_gdf)

    assert result["industrial_id"].tolist() == [2]


def test_main_rejects_invalid_approach_value(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["find_unconnected_industrial_areas.py", "--approach", "2"])

    with pytest.raises(SystemExit):
        find_unconnected_industrial_areas.main()


def test_main_rejects_multiple_approaches(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["find_unconnected_industrial_areas.py", "--approach", "0", "1"])

    with pytest.raises(SystemExit):
        find_unconnected_industrial_areas.main()


def test_main_returns_false_when_wwtps_empty(mock_cfg, monkeypatch, tmp_path):
    cfg = mock_cfg
    cfg["paths"]["industrial_unconnected_output"] = str(tmp_path / "industrial.parquet")
    cfg["overwrite_existing"] = True
    industrial_gdf = gpd.GeoDataFrame(
        {"industrial_id": [1], "geometry": [box(0.0, 0.0, 0.1, 0.1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(sys, "argv", ["find_unconnected_industrial_areas.py"])
    monkeypatch.setattr(find_unconnected_industrial_areas, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(find_unconnected_industrial_areas, "industrial_output_paths", lambda config: {"voronoi": {"1": "unused.gpkg"}})
    monkeypatch.setattr(find_unconnected_industrial_areas.os.path, "exists", lambda path: False)
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_industrial_areas", lambda config: industrial_gdf.copy())
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_wwtps", lambda config, approach_id: industrial_gdf.iloc[0:0].copy())

    assert find_unconnected_industrial_areas.main() is False


def test_main_removes_existing_output_before_parquet_write(
    mock_cfg,
    monkeypatch,
    tmp_path,
    tiny_points_gdf,
    tiny_watershed_gdf,
    tiny_country_gdf,
):
    cfg = mock_cfg
    cfg["paths"]["industrial_unconnected_output"] = str(tmp_path / "industrial.parquet")
    cfg["overwrite_existing"] = True
    cfg["prepare_data_fn"] = "prepare_data"
    captured = {}

    industrial_gdf = gpd.GeoDataFrame(
        {"industrial_id": [1], "geometry": [box(0.0, 0.0, 0.1, 0.1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    voronoi_gdf = gpd.GeoDataFrame(
        {"geometry": [box(-0.1, -0.1, 0.2, 0.2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(sys, "argv", ["find_unconnected_industrial_areas.py"])
    monkeypatch.setattr(find_unconnected_industrial_areas, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(find_unconnected_industrial_areas, "industrial_output_paths", lambda config: {"voronoi": {"1": "voronoi.gpkg"}})
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_industrial_areas", lambda config: industrial_gdf.copy())
    monkeypatch.setattr(find_unconnected_industrial_areas, "load_wwtps", lambda config, approach_id: tiny_points_gdf.iloc[:1].copy())
    monkeypatch.setattr(find_unconnected_industrial_areas, "filter_industrial_wwtps", lambda config, wwtps_gdf: wwtps_gdf.copy())
    monkeypatch.setattr(
        find_unconnected_industrial_areas,
        "_resolve_configured_callable",
        lambda *args, **kwargs: (lambda config: {"country_df": tiny_country_gdf.copy(), "basin_gdf": tiny_watershed_gdf.copy()}),
    )
    monkeypatch.setattr(find_unconnected_industrial_areas, "run_voronoi_for_wwtps", lambda *args, **kwargs: voronoi_gdf.copy())
    monkeypatch.setattr(find_unconnected_industrial_areas.os, "makedirs", lambda path, exist_ok=False: captured.setdefault("makedirs", (path, exist_ok)))
    monkeypatch.setattr(find_unconnected_industrial_areas.os, "remove", lambda path: captured.setdefault("removed", path))
    monkeypatch.setattr(
        find_unconnected_industrial_areas.os.path,
        "exists",
        lambda path: True if path == cfg["paths"]["industrial_unconnected_output"] else False,
    )

    def fake_to_parquet(self, path, index=False, **kwargs):
        captured["write"] = (path, index, len(self))

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fake_to_parquet)

    assert find_unconnected_industrial_areas.main() is True
    assert captured["removed"] == cfg["paths"]["industrial_unconnected_output"]
    assert captured["write"][0] == cfg["paths"]["industrial_unconnected_output"]
    assert captured["write"][1] is False

def test_run_voronoi_for_wwtps_reads_output_back_in_boolean_mode(monkeypatch, tmp_path):
    """Boolean mode returns (None, None) on success - the file is the evidence.

    Before this, a successful run under ``return_boolean: true`` was read as a
    failure and the script exited 1 with its output already written.
    """
    from src.industrial_analysis import find_unconnected_industrial_areas as fu

    output_path = tmp_path / "voronoi.gpkg"
    regions = gpd.GeoDataFrame(
        {"buffer_id": [1], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    regions.to_file(output_path, driver="GPKG")

    wwtps = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "site_id": [1], "geometry": [Point(0.5, 0.5)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    basins = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    cfg = {
        "basin_column_name": "HYBAS_ID",
        "country_output_column": "country",
        "site_id_column": "site_id",
        "weight_func": "mult",
        "weight_method": "linear",
        "distance_fn": lambda *a, **k: 1.0,
        "return_boolean": True,
    }

    # Boolean mode: orchestration wrote the file and reports (None, None).
    monkeypatch.setattr(fu, "run_voronoi_approach", lambda *a, **k: (None, None))

    result = fu.run_voronoi_for_wwtps(
        cfg, "1", wwtps, basins, basins, {}, str(output_path), False
    )

    assert result is not None
    assert len(result) == 1


def test_run_voronoi_for_wwtps_reports_failure_when_no_output_written(monkeypatch, tmp_path):
    from src.industrial_analysis import find_unconnected_industrial_areas as fu

    wwtps = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "site_id": [1], "geometry": [Point(0.5, 0.5)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    basins = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    cfg = {
        "basin_column_name": "HYBAS_ID",
        "country_output_column": "country",
        "site_id_column": "site_id",
        "weight_func": "mult",
        "weight_method": "linear",
        "distance_fn": lambda *a, **k: 1.0,
        "return_boolean": True,
    }
    monkeypatch.setattr(fu, "run_voronoi_approach", lambda *a, **k: (None, None))

    assert fu.run_voronoi_for_wwtps(
        cfg, "1", wwtps, basins, basins, {}, str(tmp_path / "missing.gpkg"), False
    ) is None
