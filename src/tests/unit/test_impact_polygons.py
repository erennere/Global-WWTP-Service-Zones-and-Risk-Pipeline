from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from src.pop_at_risk_river_calculations import impact_polygons_pop


pytestmark = pytest.mark.unit


def _runtime_params(impact_radii=None):
    return {
        "org_per_pop": 60.0,
        "width": 12.0,
        "c_limit": 5.0,
        "base_k": 0.23,
        "theta": 1.047,
        "step_m": 5000.0,
        "least_discharge_cms": 0.269,
        "impact_radii": impact_radii or [1000.0, 2000.0],
    }


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _ImmediateExecutor:
    def __init__(self, max_workers=None, initializer=None, initargs=()):
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self):
        if self.initializer is not None:
            self.initializer(*self.initargs)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return _ImmediateFuture(fn(*args, **kwargs))


def _set_simple_impact_globals(monkeypatch, *, start_load_ratio=10.0):
    monkeypatch.setattr(impact_polygons_pop, "next_dict", {1: (0, start_load_ratio)})
    monkeypatch.setattr(impact_polygons_pop, "geom_dict", {1: LineString([(0, 0), (100000, 0)])})
    monkeypatch.setattr(impact_polygons_pop, "lat_dict", {1: 0.0})
    monkeypatch.setattr(impact_polygons_pop, "level_dict", {1: [[1]]})
    monkeypatch.setattr(impact_polygons_pop, "discharge_dict", {1: 1.0})


def test_calculate_load_ratio_vectorized_substitutes_least_discharge():
    pop = pd.Series([100.0, 100.0])
    discharge = pd.Series([0.0, np.nan])

    result = impact_polygons_pop.calculate_load_ratio(
        pop,
        discharge,
        org_per_pop=86.4,
        c_limit=1.0,
        least_discharge_cms=0.5,
    )

    assert result.to_numpy() == pytest.approx(np.array([0.2, 0.2]))


def test_calculate_load_ratio_scalar_substitutes_least_discharge():
    result = impact_polygons_pop.calculate_load_ratio(
        100.0,
        0.0,
        org_per_pop=86.4,
        c_limit=1.0,
        least_discharge_cms=0.5,
    )

    assert result == pytest.approx(0.2)


def test_generate_single_segment_plume_area_increases_with_higher_load(monkeypatch):
    _set_simple_impact_globals(monkeypatch, start_load_ratio=10.0)

    low_polygons, low_exit = impact_polygons_pop.generate_single_segment_plume(
        1,
        lat=0.0,
        start_load_ratio=10.0,
        step_m=5000.0,
        c_limit=5.0,
        base_k=5.0,
        theta=1.0,
        impact_radii=[1000.0],
    )
    high_polygons, high_exit = impact_polygons_pop.generate_single_segment_plume(
        1,
        lat=0.0,
        start_load_ratio=40.0,
        step_m=5000.0,
        c_limit=5.0,
        base_k=5.0,
        theta=1.0,
        impact_radii=[1000.0],
    )

    assert low_polygons is not None
    assert high_polygons is not None
    assert high_polygons[0].area > low_polygons[0].area
    assert high_exit >= 0.0
    assert low_exit >= 0.0


def test_create_impact_polygons_returns_one_output_per_radius(monkeypatch):
    _set_simple_impact_globals(monkeypatch, start_load_ratio=10.0)
    pop_chunk = gpd.GeoDataFrame(
        {
            "MAIN_RIV": [1],
            "utm": [3857],
            "country": ["DE"],
            "NEXT_DOWN": [1],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = impact_polygons_pop.create_impact_polygons(
        pop_chunk,
        main_riv=1,
        nxt_dis_col="NEXT_DOWN",
        model_params=_runtime_params([500.0, 1000.0]),
    )

    assert set(result.keys()) == {500.0, 1000.0}
    assert len(result[500.0]) == 1
    assert len(result[1000.0]) == 1
    assert result[500.0].geometry.notna().all()
    assert result[1000.0].geometry.notna().all()
    assert result[1000.0].geometry.iloc[0].area > result[500.0].geometry.iloc[0].area


def test_get_runtime_params_rejects_invalid_values():
    with pytest.raises(ValueError):
        impact_polygons_pop.get_runtime_params(
            {
                "impact_polygons_pop_params": {
                    "org_per_pop": "bad",
                    "width": 12.0,
                    "c_limit": 5.0,
                    "base_k": 0.23,
                    "theta": 1.047,
                    "step_m": "250",
                    "least_discharge_cms": 0.269,
                    "impact_radii": ["300", 900],
                }
            }
        )


def test_get_runtime_params_requires_mapping_section():
    with pytest.raises(TypeError, match="impact_polygons_pop_params"):
        impact_polygons_pop.get_runtime_params({"impact_polygons_pop_params": "invalid"})


def test_create_dicts_builds_topology_and_fills_missing_weights(monkeypatch):
    monkeypatch.setattr(impact_polygons_pop, "tqdm", lambda iterable, desc=None: iterable)
    river_gdf = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1, 2, 3],
            "NEXT_DOWN": [0, 1, 2],
            "MAIN_RIV": [1, 1, 1],
            "DIS_AV_CMS": [5.0, 2.0, 1.0],
            "weight": [1.0, np.nan, 0.25],
            "lat": [0.0, 0.0, 0.0],
            "utm": [3857, 3857, 3857],
            "geometry": [
                LineString([(0.0, 0.0), (0.1, 0.0)]),
                LineString([(-0.1, 0.0), (0.0, 0.0)]),
                LineString([(-0.2, 0.0), (-0.1, 0.0)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    impact_polygons_pop.create_dicts(river_gdf, "NEXT_DOWN", "HYRIV_ID", "MAIN_RIV", "DIS_AV_CMS", "weight")

    assert impact_polygons_pop.next_dict[1] == (0, 1.0)
    assert impact_polygons_pop.next_dict[2] == (1, 0.0)
    assert impact_polygons_pop.next_dict[3] == (2, 0.25)
    assert impact_polygons_pop.discharge_dict[3] == pytest.approx(1.0)
    assert impact_polygons_pop.lat_dict[1] == pytest.approx(0.0)
    assert set(impact_polygons_pop.geom_dict.keys()) == {1, 2, 3}
    assert impact_polygons_pop.level_dict[1] == [[1], [2], [3]]


def test_orchestrate_logic_merges_radius_outputs(monkeypatch):
    monkeypatch.setattr(impact_polygons_pop, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(impact_polygons_pop, "as_completed", lambda futures: futures)
    monkeypatch.setattr(impact_polygons_pop, "tqdm", lambda iterable, total=None, desc=None: iterable)

    def fake_create_impact_polygons(pop_chunk, main_riv, nxt_dis_col, model_params=None):
        if main_riv == 1:
            poly_a = box(0.0, 0.0, 1.0, 1.0)
            poly_b = box(0.0, 0.0, 2.0, 2.0)
        else:
            poly_a = box(10.0, 10.0, 11.0, 11.0)
            poly_b = box(10.0, 10.0, 12.0, 12.0)

        return {
            500.0: gpd.GeoDataFrame(
                {"geometry": [poly_a], "utm": [3857], "country": ["DE"], "MAIN_RIV": [main_riv], nxt_dis_col: [main_riv * 10]},
                geometry="geometry",
                crs=4326,
            ),
            1000.0: gpd.GeoDataFrame(
                {"geometry": [poly_b], "utm": [3857], "country": ["DE"], "MAIN_RIV": [main_riv], nxt_dis_col: [main_riv * 10]},
                geometry="geometry",
                crs=4326,
            ),
        }

    monkeypatch.setattr(impact_polygons_pop, "create_impact_polygons", fake_create_impact_polygons)

    pop_gdf = gpd.GeoDataFrame(
        {
            "MAIN_RIV": [1, 2],
            "NEXT_DOWN": [10, 20],
            "country": ["DE", "FR"],
            "utm": [3857, 3857],
            "geometry": [Point(0, 0), Point(10, 10)],
        },
        geometry="geometry",
        crs=4326,
    )

    result = impact_polygons_pop.orchestrate_logic(
        pop_gdf,
        nxt_dis_col="NEXT_DOWN",
        main_riv_col="MAIN_RIV",
        max_workers=2,
        model_params={"impact_radii": [500.0, 1000.0]},
    )

    assert result is not None
    assert set(result.keys()) == {500.0, 1000.0}
    assert len(result[500.0]) == 2
    assert len(result[1000.0]) == 2
    assert result[500.0].geometry.is_valid.all()
    assert result[1000.0].geometry.is_valid.all()


def test_orchestrate_logic_returns_none_when_no_polygons_generated(monkeypatch):
    monkeypatch.setattr(impact_polygons_pop, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(impact_polygons_pop, "as_completed", lambda futures: futures)
    monkeypatch.setattr(impact_polygons_pop, "tqdm", lambda iterable, total=None, desc=None: iterable)
    monkeypatch.setattr(impact_polygons_pop, "create_impact_polygons", lambda *args, **kwargs: {})

    pop_gdf = gpd.GeoDataFrame(
        {
            "MAIN_RIV": [1],
            "NEXT_DOWN": [10],
            "country": ["DE"],
            "utm": [3857],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs=4326,
    )

    result = impact_polygons_pop.orchestrate_logic(pop_gdf, "NEXT_DOWN", "MAIN_RIV", max_workers=1, model_params={})

    assert result is None


def test_orchestrate_logic_rejects_non_positive_workers():
    pop_gdf = gpd.GeoDataFrame(
        {
            "MAIN_RIV": [1],
            "NEXT_DOWN": [10],
            "country": ["DE"],
            "utm": [3857],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs=4326,
    )

    with pytest.raises(ValueError, match="max_workers"):
        impact_polygons_pop.orchestrate_logic(pop_gdf, "NEXT_DOWN", "MAIN_RIV", max_workers=0, model_params={})