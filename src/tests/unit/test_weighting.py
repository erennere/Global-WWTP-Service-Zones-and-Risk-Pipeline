from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from src import create_voronoi


pytestmark = pytest.mark.unit


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


@pytest.mark.parametrize("method", ["linear", "logarithmic", "square_root", "sigmoid"])
def test_create_weights_methods_return_normalized_values(method):
    df = gpd.GeoDataFrame(
        {
            "base_values": [1.0, 4.0, 9.0],
            "geometry": [Point(0, 0), Point(1, 0), Point(2, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = create_voronoi.create_weights(df, method=method)

    assert result["weights"].between(0, 1).all()
    assert result["weights"].sum() == pytest.approx(1.0)


def test_create_weights_equal_values_stay_uniform():
    df = gpd.GeoDataFrame(
        {
            "base_values": [5.0, 5.0, 5.0],
            "geometry": [Point(0, 0), Point(1, 0), Point(2, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = create_voronoi.create_weights(df, method="linear")

    assert result["weights"].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_create_weights_zero_values_fall_back_to_equal_distribution():
    df = gpd.GeoDataFrame(
        {
            "base_values": [0.0, 0.0, 0.0],
            "geometry": [Point(0, 0), Point(1, 0), Point(2, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = create_voronoi.create_weights(df, method="linear")

    assert result["weights"].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_create_weights_single_point_returns_one():
    df = gpd.GeoDataFrame(
        {
            "base_values": [42.0],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = create_voronoi.create_weights(df, method="sigmoid")

    assert result["weights"].tolist() == pytest.approx([1.0])


def test_orchestrate_voronoi_weights_normalizes_per_basin_independently(monkeypatch, tiny_country_gdf):
    captured = []

    def fake_worker(task):
        sub_df = task[0].copy()
        captured.append(sub_df[["HYBAS_ID", "WASTE_ID", "weights"]].copy())
        region_df = gpd.GeoDataFrame(
            {"HYBAS_ID": [sub_df.iloc[0]["HYBAS_ID"]], "geometry": [sub_df.geometry.union_all().convex_hull]},
            geometry="geometry",
            crs=sub_df.crs,
        )
        point_df = sub_df[["HYBAS_ID", "WASTE_ID", "geometry"]].copy()
        return region_df, point_df

    monkeypatch.setattr(create_voronoi, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(create_voronoi, "as_completed", lambda futures: futures)
    monkeypatch.setattr(create_voronoi, "voronoi_worker", fake_worker)
    monkeypatch.setattr(create_voronoi, "estimate_utm_crs", lambda gdf: "EPSG:3857")

    df = gpd.GeoDataFrame(
        {
            "WASTE_ID": [1, 2, 3, 4],
            "HYBAS_ID": [101, 101, 202, 202],
            "ISO_2": ["DE", "DE", "FR", "FR"],
            "base_values": [1.0, 9.0, 100.0, 900.0],
            "geometry": [Point(-0.02, 0.0), Point(0.02, 0.0), Point(1.0, 0.0), Point(1.04, 0.0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    region_df, point_df = create_voronoi.orchestrate_voronoi_weights(
        df,
        "HYBAS_ID",
        tiny_country_gdf,
        workers=1,
        scale_weights=False,
        clipping=None,
        n_points=100,
        distance_fn=create_voronoi.default_distance_multiplicative,
        area_fn=lambda sub_df, **kwargs: sub_df,
        method="linear",
        calculate_buffer_fn=lambda sub_df, weights, **kwargs: np.full(len(sub_df), 1000.0),
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
    )

    assert len(region_df) == 2
    assert len(point_df) == 4
    assert len(captured) == 2

    weights_by_group = {
        str(frame["HYBAS_ID"].iloc[0]): frame.sort_values("WASTE_ID")["weights"].to_numpy()
        for frame in captured
    }
    assert weights_by_group["101"] == pytest.approx(weights_by_group["202"])
    assert weights_by_group["101"].sum() == pytest.approx(1.0)


def test_orchestrate_voronoi_weights_output_mode_resumes_from_temp_checkpoint(monkeypatch, tiny_country_gdf, tmp_path):
    output_path = tmp_path / "voronoi.gpkg"
    temp_output_path = tmp_path / "temp_voronoi.gpkg"
    state = {"temp_exists": True, "output_exists": False}
    processed_groups = []
    to_file_calls = []
    replace_calls = []

    def fake_worker(task):
        sub_df = task[0].copy()
        processed_groups.append(sub_df["HYBAS_ID"].iloc[0])
        region_df = gpd.GeoDataFrame(
            {"HYBAS_ID": [sub_df.iloc[0]["HYBAS_ID"]], "geometry": [sub_df.geometry.iloc[0]]},
            geometry="geometry",
            crs=sub_df.crs,
        )
        point_df = sub_df[["HYBAS_ID", "WASTE_ID", "geometry"]].copy()
        return region_df, point_df

    def fake_exists(path):
        if path == str(temp_output_path):
            return state["temp_exists"]
        if path == str(output_path):
            return state["output_exists"]
        return False

    def fake_read_file(path, columns=None, **kwargs):
        assert path == str(temp_output_path)
        return gpd.GeoDataFrame(
            {"HYBAS_ID": ["101"], "geometry": [Point(0, 0)]},
            geometry="geometry",
            crs="EPSG:4326",
        )

    def fake_to_file(self, filename, *args, **kwargs):
        to_file_calls.append({"filename": filename, "mode": kwargs.get("mode")})
        state["temp_exists"] = True

    def fake_replace(src, dst):
        replace_calls.append((src, dst))
        state["temp_exists"] = False
        state["output_exists"] = True

    monkeypatch.setattr(create_voronoi, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(create_voronoi, "as_completed", lambda futures: futures)
    monkeypatch.setattr(create_voronoi, "voronoi_worker", fake_worker)
    monkeypatch.setattr(create_voronoi, "estimate_utm_crs", lambda gdf: "EPSG:3857")
    monkeypatch.setattr(create_voronoi.gpd, "read_file", fake_read_file)
    monkeypatch.setattr(create_voronoi.os.path, "exists", fake_exists)
    monkeypatch.setattr(create_voronoi.os, "replace", fake_replace)
    monkeypatch.setattr(create_voronoi, "ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)

    df = gpd.GeoDataFrame(
        {
            "WASTE_ID": [1, 2],
            "HYBAS_ID": [101, 202],
            "ISO_2": ["DE", "DE"],
            "base_values": [1.0, 4.0],
            "geometry": [Point(0, 0), Point(1, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = create_voronoi.orchestrate_voronoi_weights(
        df,
        "HYBAS_ID",
        tiny_country_gdf,
        workers=1,
        area_fn=lambda sub_df, **kwargs: sub_df,
        output_path=str(output_path),
        overwrite=False,
        flush_size=1,
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
    )

    assert result is True
    assert processed_groups == ["202"]
    assert to_file_calls == [{"filename": str(temp_output_path), "mode": "a"}]
    assert replace_calls == [(str(temp_output_path), str(output_path))]


def test_orchestrate_voronoi_weights_output_mode_overwrite_clears_temp_checkpoint(monkeypatch, tiny_country_gdf, tmp_path):
    output_path = tmp_path / "voronoi.gpkg"
    temp_output_path = tmp_path / "temp_voronoi.gpkg"
    state = {"temp_exists": True, "output_exists": False}
    to_file_calls = []
    remove_calls = []
    replace_calls = []

    def fake_worker(task):
        sub_df = task[0].copy()
        region_df = gpd.GeoDataFrame(
            {"HYBAS_ID": [sub_df.iloc[0]["HYBAS_ID"]], "geometry": [sub_df.geometry.iloc[0]]},
            geometry="geometry",
            crs=sub_df.crs,
        )
        point_df = sub_df[["HYBAS_ID", "WASTE_ID", "geometry"]].copy()
        return region_df, point_df

    def fake_exists(path):
        if path == str(temp_output_path):
            return state["temp_exists"]
        if path == str(output_path):
            return state["output_exists"]
        return False

    def fail_read_file(*args, **kwargs):
        raise AssertionError("resume checkpoint should not be read when overwrite=True")

    def fake_remove(path):
        remove_calls.append(path)
        state["temp_exists"] = False

    def fake_to_file(self, filename, *args, **kwargs):
        to_file_calls.append({"filename": filename, "mode": kwargs.get("mode")})
        state["temp_exists"] = True

    def fake_replace(src, dst):
        replace_calls.append((src, dst))
        state["temp_exists"] = False
        state["output_exists"] = True

    monkeypatch.setattr(create_voronoi, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(create_voronoi, "as_completed", lambda futures: futures)
    monkeypatch.setattr(create_voronoi, "voronoi_worker", fake_worker)
    monkeypatch.setattr(create_voronoi, "estimate_utm_crs", lambda gdf: "EPSG:3857")
    monkeypatch.setattr(create_voronoi.gpd, "read_file", fail_read_file)
    monkeypatch.setattr(create_voronoi.os.path, "exists", fake_exists)
    monkeypatch.setattr(create_voronoi.os, "remove", fake_remove)
    monkeypatch.setattr(create_voronoi.os, "replace", fake_replace)
    monkeypatch.setattr(create_voronoi, "ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)

    df = gpd.GeoDataFrame(
        {
            "WASTE_ID": [1],
            "HYBAS_ID": [101],
            "ISO_2": ["DE"],
            "base_values": [1.0],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = create_voronoi.orchestrate_voronoi_weights(
        df,
        "HYBAS_ID",
        tiny_country_gdf,
        workers=1,
        area_fn=lambda sub_df, **kwargs: sub_df,
        output_path=str(output_path),
        overwrite=True,
        flush_size=1,
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
    )

    assert result is True
    assert remove_calls == [str(temp_output_path)]
    assert to_file_calls == [{"filename": str(temp_output_path), "mode": None}]
    assert replace_calls == [(str(temp_output_path), str(output_path))]


def test_initialize_voronoi_weights_additive_mode_pins_factor_to_one():
    """Additive scaling is pinned to 1.0, matching the frozen baseline tree.

    ``auto_weight_scale`` is deliberately left uncalled there (the call site is
    commented out), so weights pass through unscaled. This guards that parity.
    """
    df = gpd.GeoDataFrame(
        {
            "weights": [0.25, 0.75],
            "geometry": [Point(0, 0), Point(1, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    weights, factor = create_voronoi.initialize_voronoi_weights(
        df,
        create_voronoi.default_distance_additive,
        scale_weights=True,
        points=[(0, 0), (1, 0)],
    )

    assert factor == pytest.approx(1.0)
    assert weights == pytest.approx(np.array([0.25, 0.75]))