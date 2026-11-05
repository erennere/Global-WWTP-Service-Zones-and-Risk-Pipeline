import os
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from src import pipelines


pytestmark = pytest.mark.unit


def named_fn():
    return "named"


def test_resolve_configured_callable_handles_default_and_named_functions():
    module = SimpleNamespace(__name__="dummy_module", named_fn=named_fn)

    assert pipelines._resolve_configured_callable(None, named_fn, "callable_key", module) is named_fn
    assert pipelines._resolve_configured_callable("   ", named_fn, "callable_key", module) is named_fn
    assert pipelines._resolve_configured_callable("named_fn", named_fn, "callable_key", module) is named_fn


def test_resolve_configured_callable_rejects_missing_and_invalid_types():
    module = SimpleNamespace(__name__="dummy_module")

    with pytest.raises(ValueError, match="references 'missing_fn'"):
        pipelines._resolve_configured_callable("missing_fn", named_fn, "callable_key", module)

    with pytest.raises(TypeError, match="callable or string"):
        pipelines._resolve_configured_callable(123, named_fn, "callable_key", module)


def test_create_output_paths_and_pop_output_paths_use_config_tokens(tmp_path):
    cfg = {
        "version": "2",
        "level": "7",
        "buffer_path_token": "q0_6",
        "weight_func": "",
        "weight_func_suffix": "",
        "weight_type": "log",
        "paths": {
            "buffers_dir": str(tmp_path / "buffers"),
            "voronoi_dir": str(tmp_path / "voronoi" / "log"),
            "pop_output_dir": str(tmp_path / "pop_voronoi" / "log"),
        },
    }

    output_paths = pipelines.create_output_paths(cfg)
    pop_output_paths = pipelines.create_pop_output_paths(cfg)

    assert Path(output_paths["buffers"]["WWTP"]).name == "dissolved_wwtp_buffers_v2_lvl7_bfq0_6.gpkg"
    assert Path(output_paths["voronoi"]["1"]).name == "appr_1_v2_lvl7_bfq0_6_log.gpkg"
    assert Path(pop_output_paths["voronoi"]["1"]).name == "pop_added_appr_1_v2_lvl7_bfq0_6_log.gpkg"


def _minimal_run_cfg(**overrides):
    cfg = {
        "overwrite_existing": False,
        "country_output_column": "ISO_2",
        "country_boundary_column": "country",
        "site_id_column": "WASTE_ID",
        "calculate_buffer_kwargs": {"buffer": 1000},
        "calculate_area_fn": None,
        "calculate_buffer_fn": None,
        "area_fn_kwargs": {},
        "max_workers": 2,
        "n_points": 10,
        "scipy_true": False,
        "cv2_true": False,
        "threshold": 500,
        "sigma": 3,
        "percent_threshold": 10,
        "return_boolean": True,
        "temp_voronoi_overwrite": False,
        "flush_size": 100,
    }
    cfg.update(overrides)
    return cfg


def _sample_run_inputs():
    gdf = gpd.GeoDataFrame(
        {"WASTE_ID": [1], "ISO_2": ["DE"], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    clipping_gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0).buffer(0.1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    country_df = gpd.GeoDataFrame(
        {"country": ["DE"], "geometry": [Point(0, 0).buffer(1.0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    return gdf, clipping_gdf, country_df


def test_run_voronoi_approach_skips_existing_output(tmp_path):
    output_path = tmp_path / "existing.gpkg"
    output_path.write_text("present", encoding="utf-8")
    gdf, clipping_gdf, country_df = _sample_run_inputs()

    result = pipelines.run_voronoi_approach(
        "0",
        gdf,
        clipping_gdf,
        country_df,
        _minimal_run_cfg(),
        distance_fn=lambda *args, **kwargs: None,
        output_path=str(output_path),
    )

    assert result == (None, None)


def test_run_voronoi_approach_passes_resolved_functions_and_only_round(monkeypatch, tmp_path):
    captured = {}

    def fake_orchestrate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr("src.service_areas.orchestrate_voronoi_weights", fake_orchestrate)

    gdf, clipping_gdf, country_df = _sample_run_inputs()
    output_path = tmp_path / "result.gpkg"

    result = pipelines.run_voronoi_approach(
        "1",
        gdf,
        clipping_gdf,
        country_df,
        _minimal_run_cfg(
            calculate_area_fn="calculate_area",
            calculate_buffer_fn="calculate_buffer",
            area_fn_kwargs={"custom": "value"},
        ),
        distance_fn="distance-sentinel",
        output_path=str(output_path),
        only_round=True,
        buffering=True,
        method="logarithmic",
    )

    assert result == (None, None)
    assert captured["args"][0] is gdf
    assert captured["args"][1] == "buffer_id"
    assert captured["kwargs"]["clipping"] is clipping_gdf
    assert captured["kwargs"]["distance_fn"] == "distance-sentinel"
    assert captured["kwargs"]["buffering"] is True
    assert captured["kwargs"]["method"] == "logarithmic"
    assert captured["kwargs"]["output_path"] == str(output_path)
    assert captured["kwargs"]["area_fn"].__name__ == "calculate_area"
    assert captured["kwargs"]["calculate_buffer_fn"].__name__ == "calculate_buffer"
    assert captured["kwargs"]["area_fn_kwargs"] == {"custom": "value", "only_round": True}


def test_run_voronoi_approach_writes_tuple_result(monkeypatch, tmp_path):
    region_df = gpd.GeoDataFrame(
        {"region_id": [1], "geometry": [Point(0, 0).buffer(0.5)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    point_df = gpd.GeoDataFrame(
        {"WASTE_ID": [1], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    captured = {}

    monkeypatch.setattr(
        "src.service_areas.orchestrate_voronoi_weights",
        lambda *args, **kwargs: (region_df, point_df),
    )
    monkeypatch.setattr(
        "src.utils.ensure_output_dir_for_file",
        lambda path: captured.setdefault("ensured_path", path),
    )

    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, path, driver=None, index=None, **kwargs):
        captured["written_path"] = path
        captured["driver"] = driver
        captured["index"] = index

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)

    gdf, clipping_gdf, country_df = _sample_run_inputs()
    output_path = tmp_path / "tuple_result.gpkg"
    result = pipelines.run_voronoi_approach(
        "2",
        gdf,
        clipping_gdf,
        country_df,
        _minimal_run_cfg(return_boolean=False),
        distance_fn=lambda *args, **kwargs: None,
        output_path=str(output_path),
    )

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert result == (region_df, point_df)
    assert captured["ensured_path"] == str(output_path)
    assert captured["written_path"] == str(output_path)
    assert captured["driver"] == "GPKG"
    assert captured["index"] is False


def test_run_voronoi_approach_rejects_invalid_area_kwargs_type(tmp_path):
    gdf, clipping_gdf, country_df = _sample_run_inputs()

    with pytest.raises(TypeError, match="area_fn_kwargs"):
        pipelines.run_voronoi_approach(
            "0",
            gdf,
            clipping_gdf,
            country_df,
            _minimal_run_cfg(area_fn_kwargs="bad"),
            distance_fn=lambda *args, **kwargs: None,
            output_path=str(tmp_path / "invalid.gpkg"),
        )


def test_run_voronoi_approach_rejects_non_positive_max_workers(tmp_path):
    gdf, clipping_gdf, country_df = _sample_run_inputs()

    with pytest.raises(ValueError, match="max_workers"):
        pipelines.run_voronoi_approach(
            "0",
            gdf,
            clipping_gdf,
            country_df,
            _minimal_run_cfg(max_workers=0),
            distance_fn=lambda *args, **kwargs: None,
            output_path=str(tmp_path / "invalid_workers.gpkg"),
        )


def test_prepare_data_uses_final_geometry_and_filters_industrial_sites(monkeypatch, tmp_path):
    rng = np.random.default_rng(101)
    cfg = {
        "csv_files": False,
        "paths": {
            "annotated_all_filepath": str(tmp_path / "corrected_all.gpkg"),
            "watershed": str(tmp_path / "watersheds.geojson"),
            "overture": str(tmp_path / "overture.parquet"),
            "overture_s3_url": "s3://bucket/overture.parquet",
            "bboxes": str(tmp_path / "bboxes.csv"),
        },
        "country_output_column": "ISO_2",
        "country_boundary_column": "country",
        "site_id_column": "WASTE_ID",
        "old_site_id_column": "old_WASTE_ID",
        "basin_column_name": "HYBAS_ID",
        "remove_industrial": True,
        "force_country_rejoin": True,
        "industrial_category_numbers": ["3"],
        "mix_use_categories": [],
        "sindex_concurrency": False,
    }

    points = [Point(*rng.random(2)), Point(*(rng.random(2) + 1)), Point(*(rng.random(2) + 2))]
    final_points = [Point(pt.x + 0.1, pt.y + 0.1) for pt in points]
    bbox_gdf = gpd.GeoDataFrame(
        {
            "WASTE_ID": [10, 20, 30],
            "HYBAS_ID": [101, 202, 101],
            "category_number": ["1", "3", "mix"],
            "ISO_2": ["old", "old", "old"],
            "geometry": points,
            "final_geometry": final_points,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    basin_gdf = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 202],
            "ISO_2": ["stale", "stale"],
            "geometry": [Point(0, 0).buffer(1.0), Point(2, 2).buffer(1.0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    country_df = pd.DataFrame(
        {
            "country": ["DE", "FR"],
            "geometry": [geom.wkb for geom in basin_gdf.geometry],
        }
    )

    def fake_read_file(path, crs=None):
        if path == cfg["paths"]["annotated_all_filepath"]:
            return bbox_gdf.copy()
        if path == cfg["paths"]["watershed"]:
            return basin_gdf.copy()
        raise AssertionError(f"Unexpected read path: {path}")

    def fake_intersects_with_country_db(gdf, path, polygon_country_col=None, output_country_col=None):
        if "WASTE_ID" in gdf.columns:
            return gdf.assign(**{output_country_col: ["DE"] * len(gdf)})
        return gdf.assign(**{output_country_col: ["DE", "FR"][: len(gdf)]})

    monkeypatch.setattr("src.service_areas.drop_duplicates", lambda gdf, col: gdf)
    monkeypatch.setattr("src.geo_utils.buffer_geometry", lambda geom: geom)
    monkeypatch.setattr("src.service_areas.intersects_with_country_db", fake_intersects_with_country_db)
    monkeypatch.setattr("src.service_areas.download_overture_maps", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("download should not run when overture exists")))
    monkeypatch.setattr("src.service_areas.intersect_with_polygon_sindex", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("basin join should not run when HYBAS_ID already exists")))
    monkeypatch.setattr("src.utils.ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(pipelines.gpd, "read_file", fake_read_file)
    monkeypatch.setattr(pipelines.pd, "read_parquet", lambda path: country_df.copy())
    monkeypatch.setattr(
        pipelines.os.path,
        "exists",
        lambda path: path in {cfg["paths"]["overture"], os.path.abspath(cfg["paths"]["watershed"].replace('.geojson', '.gpkg'))},
    )

    result = pipelines.prepare_data(cfg)

    gdf_bbox = result["gdf_bbox"].sort_values("old_WASTE_ID").reset_index(drop=True)
    # R12: IDs are assigned once over all plants, before remove_industrial, so the
    # kept plants carry the IDs they would have without the filter.
    all_plants = bbox_gdf.assign(geometry=final_points).drop(columns=["final_geometry"])
    all_plants["WKT_WWTP"] = all_plants.geometry.to_wkt()
    expected = pipelines.assign_stable_site_ids(all_plants, "WASTE_ID", "old_WASTE_ID").set_index("old_WASTE_ID")
    assert gdf_bbox["old_WASTE_ID"].tolist() == [10, 30]
    assert gdf_bbox["WASTE_ID"].tolist() == expected.loc[[10, 30], "WASTE_ID"].tolist()
    assert gdf_bbox["SITE_UID"].tolist() == expected.loc[[10, 30], "SITE_UID"].tolist()
    assert gdf_bbox["geometry"].tolist() == [final_points[0], final_points[2]]
    assert gdf_bbox["ISO_2"].tolist() == ["DE", "DE"]
    assert "final_geometry" not in gdf_bbox.columns
    assert gdf_bbox["WKT_WWTP"].notna().all()
    assert result["basin_gdf"]["basin_area"].gt(0).all()
    assert result["country_df"].crs.to_epsg() == 4326


def test_prepare_data_csv_mode_downloads_overture_once_and_exports_expanded_outputs(monkeypatch, tmp_path):
    rng = np.random.default_rng(202)
    cfg = {
        "csv_files": True,
        "paths": {
            "bboxes": str(tmp_path / "bboxes.csv"),
            "hydrowaste": str(tmp_path / "hydrowaste.csv"),
            "watershed": str(tmp_path / "watersheds.geojson"),
            "overture": str(tmp_path / "overture.parquet"),
            "overture_s3_url": "s3://bucket/overture.parquet",
        },
        "country_output_column": "ISO_2",
        "country_boundary_column": "country",
        "site_id_column": "WASTE_ID",
        "old_site_id_column": "old_WASTE_ID",
        "basin_column_name": "HYBAS_ID",
        "remove_industrial": False,
        "force_country_rejoin": True,
        "industrial_category_numbers": ["3"],
        "mix_use_categories": [],
        "sindex_concurrency": True,
    }
    state = {
        "overture_exists": False,
        "watershed_gpkg_exists": False,
        "expanded_csv_exists": False,
        "expanded_gpkg_exists": False,
    }
    captured = {"to_file": [], "to_csv": []}

    bbox_df = pd.DataFrame(
        {
            "WASTE_ID": [1, 2],
            "geometry": [Point(*rng.random(2)).wkt, Point(*(rng.random(2) + 1)).wkt],
        }
    )
    hydrowaste_df = pd.DataFrame(
        {
            "WASTE_ID": [1, 2],
            "LON_WWTP": [0.0, 1.0],
            "LAT_WWTP": [0.0, 1.0],
            "geometry": ["POINT (0 0)", "POINT (1 1)"],
            "POP_SERVED": [100, 200],
            "extra_attr": ["a", "b"],
        }
    )
    basin_gdf = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 202],
            "geometry": [Point(0, 0).buffer(1.0), Point(2, 2).buffer(1.0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    country_df = pd.DataFrame(
        {
            "country": ["DE", "FR"],
            "geometry": [geom.wkb for geom in basin_gdf.geometry],
        }
    )

    def fake_read_csv(path):
        if path == cfg["paths"]["bboxes"]:
            return bbox_df.copy()
        if path == cfg["paths"]["hydrowaste"]:
            return hydrowaste_df.copy()
        raise AssertionError(f"Unexpected CSV path: {path}")

    def fake_read_file(path, crs=None):
        if path == cfg["paths"]["watershed"]:
            return basin_gdf.copy()
        raise AssertionError(f"Unexpected read path: {path}")

    def fake_exists(path):
        watershed_gpkg = os.path.abspath(cfg["paths"]["watershed"].replace('.geojson', '.gpkg'))
        expanded_csv = os.path.join(os.path.dirname(cfg["paths"]["bboxes"]), f"expanded_{os.path.basename(cfg['paths']['bboxes'])}")
        expanded_gpkg = expanded_csv.replace('.csv', '.gpkg')
        if path == cfg["paths"]["overture"]:
            return state["overture_exists"]
        if path == watershed_gpkg:
            return state["watershed_gpkg_exists"]
        if path == expanded_csv:
            return state["expanded_csv_exists"]
        if path == expanded_gpkg:
            return state["expanded_gpkg_exists"]
        return False

    def fake_download(url, path):
        state["overture_exists"] = True

    def fake_intersects_with_country_db(gdf, path, polygon_country_col=None, output_country_col=None):
        return gdf.assign(**{output_country_col: ["DE", "FR"][: len(gdf)]})

    def fake_intersect_with_polygon_sindex(gdf, basin_slice, basin_col, concurrency=False):
        return gdf.assign(**{basin_col: [101, 202]})

    def fake_to_csv(self, path, index=False, **kwargs):
        captured["to_csv"].append(path)
        state["expanded_csv_exists"] = True

    def fake_to_file(self, path, *args, **kwargs):
        captured["to_file"].append(path)
        if path.endswith("watersheds.gpkg"):
            state["watershed_gpkg_exists"] = True
        if path.endswith("expanded_bboxes.gpkg"):
            state["expanded_gpkg_exists"] = True

    monkeypatch.setattr("src.service_areas.drop_duplicates", lambda gdf, col: gdf)
    monkeypatch.setattr("src.geo_utils.buffer_geometry", lambda geom: geom)
    monkeypatch.setattr("src.service_areas.download_overture_maps", fake_download)
    monkeypatch.setattr("src.service_areas.intersects_with_country_db", fake_intersects_with_country_db)
    monkeypatch.setattr("src.service_areas.intersect_with_polygon_sindex", fake_intersect_with_polygon_sindex)
    monkeypatch.setattr("src.utils.ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(pipelines.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(pipelines.gpd, "read_file", fake_read_file)
    monkeypatch.setattr(pipelines.pd, "read_parquet", lambda path: country_df.copy())
    monkeypatch.setattr(pipelines.os.path, "exists", fake_exists)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)

    result = pipelines.prepare_data(cfg)

    expanded_csv = os.path.join(os.path.dirname(cfg["paths"]["bboxes"]), "expanded_bboxes.csv")
    expanded_gpkg = expanded_csv.replace('.csv', '.gpkg')
    watershed_gpkg = os.path.abspath(cfg["paths"]["watershed"].replace('.geojson', '.gpkg'))
    assert state["overture_exists"] is True
    assert expanded_csv in captured["to_csv"]
    assert watershed_gpkg in captured["to_file"]
    assert expanded_gpkg in captured["to_file"]
    assert result["gdf_bbox"]["HYBAS_ID"].tolist() == [101, 202]
    assert result["gdf_bbox"]["basin_area"].notna().all()
    assert result["country_df"].crs.to_epsg() == 4326