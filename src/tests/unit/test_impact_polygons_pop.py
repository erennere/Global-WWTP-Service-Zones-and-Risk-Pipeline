from __future__ import annotations

import runpy
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from src.pop_at_risk_river_calculations import impact_polygons_pop as ipp


pytestmark = pytest.mark.unit


def _runtime_params(impact_radii=None):
    return {
        "org_per_pop": 60.0,
        "width": 12.0,
        "c_limit": 5.0,
        "base_k": 0.23,
        "theta": 1.047,
        "step_m": 100.0,
        "least_discharge_cms": 0.269,
        "velocity_m_s": 1.0,
        "water_temp_equator_c": 28.0,
        "impact_radii": impact_radii or [1000.0, 2000.0],
    }


class _ImmediateFuture:
    def __init__(self, fn, args, kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def result(self):
        return self._fn(*self._args, **self._kwargs)


class _ImmediateExecutor:
    def __init__(self, max_workers=None, initializer=None, initargs=()):
        self.max_workers = max_workers
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return _ImmediateFuture(fn, args, kwargs)


def _patch_immediate_executor(monkeypatch):
    monkeypatch.setattr(ipp, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(ipp, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(ipp, "tqdm", lambda iterable, **kwargs: iterable)


def _single_polygon_result(main_riv=1, hyriv_id=1):
    return gpd.GeoDataFrame(
        {
            "geometry": [box(0, 0, 1, 1)],
            "utm": [32632],
            "country": ["DE"],
            "MAIN_RIV": [main_riv],
            "HYRIV_ID": [hyriv_id],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


@pytest.fixture(autouse=True)
def _reset_globals():
    original = {
        "next_dict": ipp.next_dict,
        "geom_dict": ipp.geom_dict,
        "lat_dict": ipp.lat_dict,
        "level_dict": ipp.level_dict,
        "discharge_dict": ipp.discharge_dict,
    }
    yield
    ipp.next_dict = original["next_dict"]
    ipp.geom_dict = original["geom_dict"]
    ipp.lat_dict = original["lat_dict"]
    ipp.level_dict = original["level_dict"]
    ipp.discharge_dict = original["discharge_dict"]


def test_import_fallback_block_executes():
    module_path = Path(__file__).resolve().parents[3] / "src" / "pop_at_risk_river_calculations" / "impact_polygons_pop.py"

    module_globals = runpy.run_path(str(module_path), run_name="not_main")

    assert "create_impact_polygons" in module_globals


def test_batch_estimate_utm_epsg_marks_invalid_coordinates():
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(10, 50), Point(200, 90)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    epsg_codes, lats = ipp.batch_estimate_utm_epsg(gdf)

    assert list(lats) == [50.0, 90.0]
    assert epsg_codes[0] != 3857
    assert epsg_codes[1] == 3857


def test_calculate_load_ratio_handles_numpy_zero_and_nan_discharge():
    pop = np.array([10.0, 15.0])
    discharge = np.array([0.0, np.nan])

    result = ipp.calculate_load_ratio(pop, discharge, c_limit=5.0)

    assert np.isfinite(result).all()
    assert (result > 0).all()


def test_generate_single_segment_plume_handles_missing_and_short_segments():
    ipp.next_dict = {}
    ipp.geom_dict = {}
    ipp.discharge_dict = {}

    assert ipp.generate_single_segment_plume(999, lat=50.0) == (None, 0.0)

    ipp.next_dict = {1: (0, 0.5)}
    ipp.geom_dict = {1: LineString([(0.0, 0.0), (0.0001, 0.0001)])}
    ipp.discharge_dict = {1: 1.0}

    assert ipp.generate_single_segment_plume(1, lat=50.0) == (None, 0.0)


def test_generate_single_segment_plume_ignores_polygon_creation_errors(monkeypatch):
    ipp.next_dict = {1: (0, 10.0)}
    ipp.geom_dict = {1: LineString([(0.0, 0.0), (200.0, 0.0)])}
    ipp.discharge_dict = {1: 1.0}

    monkeypatch.setattr(ipp, "Polygon", lambda coords: (_ for _ in ()).throw(ValueError("bad polygon")))

    polygons, exit_load = ipp.generate_single_segment_plume(1, lat=5.0, c_limit=1.0, step_m=100.0, impact_radii=[1000.0])

    assert polygons is None
    assert exit_load == 0.0


def test_generate_single_segment_plume_uses_normalized_threshold_without_second_c_limit_division():
    ipp.next_dict = {1: (0, 4.0)}
    ipp.geom_dict = {1: LineString([(0.0, 0.0), (200.0, 0.0)])}
    ipp.discharge_dict = {1: 1.0}

    polygons, exit_load = ipp.generate_single_segment_plume(
        1,
        lat=5.0,
        c_limit=5.0,
        step_m=100.0,
        impact_radii=[1000.0],
    )

    assert polygons is not None
    assert len(polygons) == 1
    assert polygons[0].is_valid
    assert exit_load >= 0.0


def test_generate_single_segment_plume_uses_provided_local_next_state_dict(monkeypatch):
    ipp.next_dict = {}
    ipp.geom_dict = {1: LineString([(0.0, 0.0), (200.0, 0.0)])}
    ipp.discharge_dict = {1: 1.0}

    local_next_dict = {1: (0, 4.0)}
    polygons, exit_load = ipp.generate_single_segment_plume(
        1,
        lat=5.0,
        start_load_ratio=None,
        c_limit=5.0,
        step_m=100.0,
        impact_radii=[1000.0],
        next_state_dict=local_next_dict,
    )

    assert polygons is not None
    assert len(polygons) == 1
    assert polygons[0].is_valid
    assert exit_load >= 0.0


def test_create_impact_polygons_returns_empty_for_empty_chunk_and_missing_levels():
    empty = gpd.GeoDataFrame({"HYRIV_ID": [], "geometry": []}, geometry="geometry", crs="EPSG:4326")
    assert ipp.create_impact_polygons(empty, 1, "HYRIV_ID") == {}

    pop_chunk = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1],
            "utm": [32632],
            "country": ["DE"],
            "MAIN_RIV": [10],
            "geometry": [Point(10, 50)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    ipp.level_dict = {}
    ipp.next_dict = {1: (0, 2.0)}
    ipp.geom_dict = {1: LineString([(0.0, 0.0), (1.0, 0.0)])}
    ipp.discharge_dict = {1: 1.0}

    assert ipp.create_impact_polygons(pop_chunk, 10, "HYRIV_ID") == {}


def test_create_impact_polygons_handles_state_branches_and_downstream_propagation(monkeypatch):
    pop_chunk = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1, 2],
            "utm": [32632, 32632],
            "country": ["DE", "DE"],
            "MAIN_RIV": [10, 10],
            "geometry": [Point(10, 50), Point(10.1, 50.1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    ipp.level_dict = {10: [[1, 2, 3]]}
    ipp.next_dict = {1: (2, 2.0), 2: (0, 0.0)}
    ipp.lat_dict = {1: 50.0}
    ipp.discharge_dict = {2: 1.0}

    def fake_generate_single_segment_plume(rid, lat, **kwargs):
        if rid == 1:
            return ([box(0, 0, 1, 1), box(0, 0, 2, 2)], 5.0)
        if rid == 2:
            return (None, 0.0)
        pytest.fail(f"unexpected rid {rid}")

    monkeypatch.setattr(ipp, "generate_single_segment_plume", fake_generate_single_segment_plume)

    result = ipp.create_impact_polygons(
        pop_chunk,
        10,
        "HYRIV_ID",
        model_params=_runtime_params([1000.0, 2000.0]),
    )

    assert set(result) == {1000.0, 2000.0}
    assert len(result[1000.0]) == 1
    assert len(result[2000.0]) == 1


def test_create_impact_polygons_skips_zero_load_non_target_segment(monkeypatch):
    pop_chunk = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [2],
            "utm": [32632],
            "country": ["DE"],
            "MAIN_RIV": [10],
            "geometry": [Point(10.1, 50.1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    ipp.level_dict = {10: [[1, 2]]}
    ipp.next_dict = {1: (0, 0.0), 2: (0, 2.0)}
    ipp.lat_dict = {2: 50.0}
    ipp.discharge_dict = {2: 1.0}
    seen = []

    def fake_generate_single_segment_plume(rid, lat, **kwargs):
        seen.append(rid)
        return ([box(0, 0, 1, 1)], 0.0)

    monkeypatch.setattr(ipp, "generate_single_segment_plume", fake_generate_single_segment_plume)

    result = ipp.create_impact_polygons(
        pop_chunk,
        10,
        "HYRIV_ID",
        model_params=_runtime_params([1000.0]),
    )

    assert seen == [2]
    assert set(result) == {1000.0}
    assert len(result[1000.0]) == 1


def test_create_impact_polygons_returns_empty_on_exception(monkeypatch):
    pop_chunk = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1],
            "utm": [32632],
            "country": ["DE"],
            "MAIN_RIV": [10],
            "geometry": [Point(10, 50)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    ipp.level_dict = {10: [[1]]}
    ipp.next_dict = {1: (0, 2.0)}
    ipp.lat_dict = {1: 50.0}
    ipp.discharge_dict = {1: 1.0}

    monkeypatch.setattr(ipp, "generate_single_segment_plume", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    assert ipp.create_impact_polygons(pop_chunk, 10, "HYRIV_ID", model_params=_runtime_params()) == {}


def test_parallel_dissolve_returns_empty_gdf_for_empty_subset():
    subset = pd.DataFrame(columns=["geometry"])

    result = ipp.parallel_dissolve(subset, 32632)

    assert result.empty


def test_orchestrate_logic_handles_stage1_worker_failures(monkeypatch):
    pop_gdf = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1, 2],
            "MAIN_RIV": [1, 2],
            "utm": [32632, 32632],
            "country": ["DE", "DE"],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    def fake_create_impact_polygons(pop_chunk, main_riv, nxt_dis_col, model_params=None):
        if main_riv == 1:
            raise RuntimeError("plume failure")
        return {1000.0: _single_polygon_result(main_riv=main_riv, hyriv_id=int(pop_chunk["HYRIV_ID"].iloc[0]))}

    _patch_immediate_executor(monkeypatch)
    monkeypatch.setattr(ipp, "create_impact_polygons", fake_create_impact_polygons)
    monkeypatch.setattr(
        ipp,
        "parallel_dissolve",
        lambda subset_df, crs_code: _single_polygon_result(),
    )

    result = ipp.orchestrate_logic(pop_gdf, "HYRIV_ID", "MAIN_RIV", max_workers=1)

    assert 1000.0 in result


def test_orchestrate_logic_returns_none_when_dissolve_workers_fail(monkeypatch):
    pop_gdf = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1],
            "MAIN_RIV": [1],
            "utm": [32632],
            "country": ["DE"],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    _patch_immediate_executor(monkeypatch)
    monkeypatch.setattr(
        ipp,
        "create_impact_polygons",
        lambda *args, **kwargs: {1000.0: _single_polygon_result()},
    )
    monkeypatch.setattr(ipp, "parallel_dissolve", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("dissolve failed")))

    assert ipp.orchestrate_logic(pop_gdf, "HYRIV_ID", "MAIN_RIV", max_workers=1) is None


def test_orchestrate_logic_falls_back_to_unary_union_when_standard_dissolve_fails(monkeypatch):
    pop_gdf = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1],
            "MAIN_RIV": [1],
            "utm": [32632],
            "country": ["DE"],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    _patch_immediate_executor(monkeypatch)
    monkeypatch.setattr(
        ipp,
        "create_impact_polygons",
        lambda *args, **kwargs: {1000.0: _single_polygon_result()},
    )
    monkeypatch.setattr(
        ipp,
        "parallel_dissolve",
        lambda subset_df, crs_code: _single_polygon_result(),
    )
    monkeypatch.setattr(gpd.GeoDataFrame, "dissolve", lambda self, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad dissolve")))

    result = ipp.orchestrate_logic(pop_gdf, "HYRIV_ID", "MAIN_RIV", max_workers=1)

    assert 1000.0 in result
    assert not result[1000.0].empty


def test_main_rejects_non_positive_workers(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "non_served_nxt_river_outpath": str(tmp_path / "non_served.gpkg"),
            "rivershed_output_path": str(tmp_path / "rivers.gpkg"),
            "impact_pop_polygons_outpath": str(tmp_path / "impact.gpkg"),
        },
        "impact_polygons_pop_params": _runtime_params(),
    }
    pop_gdf = gpd.GeoDataFrame(
        {
            "NXT_DIS": [1],
            "pop_sum": [10],
            "country": ["DE"],
            "utm": [32632],
            "geometry": [Point(10, 50)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    river_gdf = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1],
            "NEXT_DOWN": [0],
            "MAIN_RIV": [1],
            "DIS_AV_CMS": [1.0],
            "geometry": [LineString([(10, 50), (10.1, 50.1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(ipp.os, "chdir", lambda path: None)
    monkeypatch.setattr(ipp, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(ipp, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(
        ipp.gpd,
        "read_file",
        lambda path, columns=None, **kwargs: pop_gdf.copy() if str(path).endswith("non_served.gpkg") else river_gdf.copy(),
    )
    monkeypatch.setattr(ipp, "batch_estimate_utm_epsg", lambda gdf: (np.array([32632]), np.array([50.0])))
    monkeypatch.setattr(ipp, "create_dicts", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["prog", "--max-workers", "0"])

    with pytest.raises(ValueError, match="max_workers"):
        ipp.main()


def _radius_override_main(monkeypatch, tmp_path, argv):
    """Run ipp.main() with the plumbing stubbed, returning the radii it used.

    Shared by the --radius cases below: everything downstream of the override
    is faked out so the assertion is purely about which radii reach
    orchestrate_logic and which files get written.
    """
    cfg = {
        "paths": {
            "non_served_nxt_river_outpath": str(tmp_path / "non_served.gpkg"),
            "rivershed_output_path": str(tmp_path / "rivers.gpkg"),
            "impact_pop_polygons_outpath": str(tmp_path / "impact.gpkg"),
        },
        "impact_polygons_pop_params": _runtime_params(),
    }
    pop_gdf = gpd.GeoDataFrame(
        {
            "NXT_DIS": [1],
            "pop_sum": [10],
            "country": ["DE"],
            "utm": [32632],
            "geometry": [Point(10, 50)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    river_gdf = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1],
            "NEXT_DOWN": [0],
            "MAIN_RIV": [1],
            "DIS_AV_CMS": [1.0],
            "geometry": [LineString([(10, 50), (10.1, 50.1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    seen = {}
    written = []

    def fake_orchestrate(pop, id_col, main_col, max_workers=1, model_params=None):
        seen["impact_radii"] = list(model_params["impact_radii"])
        return {
            radius: gpd.GeoDataFrame(
                {"geometry": [box(0, 0, 1, 1)]}, geometry="geometry", crs="EPSG:4326"
            )
            for radius in model_params["impact_radii"]
        }

    monkeypatch.setattr(ipp.os, "chdir", lambda path: None)
    monkeypatch.setattr(ipp, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(ipp, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(
        ipp.gpd,
        "read_file",
        lambda path, columns=None, **kwargs: pop_gdf.copy() if str(path).endswith("non_served.gpkg") else river_gdf.copy(),
    )
    monkeypatch.setattr(ipp, "batch_estimate_utm_epsg", lambda gdf: (np.array([32632]), np.array([50.0])))
    monkeypatch.setattr(ipp, "create_dicts", lambda *args, **kwargs: None)
    monkeypatch.setattr(ipp, "orchestrate_logic", fake_orchestrate)
    monkeypatch.setattr(ipp, "ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(
        gpd.GeoDataFrame, "to_file", lambda self, path, driver="GPKG": written.append(str(path))
    )
    monkeypatch.setattr(sys, "argv", argv)

    ipp.main()
    return cfg, seen, written


def test_main_without_radius_processes_every_configured_radius(monkeypatch, tmp_path):
    """Unset --radius leaves the configured radii list untouched."""
    cfg, seen, written = _radius_override_main(monkeypatch, tmp_path, ["prog"])

    assert seen["impact_radii"] == [1000.0, 2000.0]
    assert sorted(written) == sorted(
        ipp.impact_polygons_output_path(cfg["paths"]["impact_pop_polygons_outpath"], r)
        for r in (1000.0, 2000.0)
    )


def test_main_with_radius_processes_only_that_radius(monkeypatch, tmp_path):
    """--radius restricts the run to one radius and one output file."""
    cfg, seen, written = _radius_override_main(
        monkeypatch, tmp_path, ["prog", "--radius", "2000"]
    )

    assert seen["impact_radii"] == [2000.0]
    assert written == [
        ipp.impact_polygons_output_path(cfg["paths"]["impact_pop_polygons_outpath"], 2000.0)
    ]
    # The helper is what main() uses, so this pins the actual on-disk name too.
    assert written[0].endswith("impact_2000.gpkg")


def test_main_rejects_radius_not_in_configured_list(monkeypatch, tmp_path):
    """An unconfigured --radius fails loudly instead of silently doing nothing."""
    with pytest.raises(ValueError, match="Unknown --radius"):
        _radius_override_main(monkeypatch, tmp_path, ["prog", "--radius", "1234"])


def test_impact_polygons_script_entrypoint_runs_main_guard(monkeypatch):
    import src.starter as starter_mod

    module_path = Path(__file__).resolve().parents[3] / "src" / "pop_at_risk_river_calculations" / "impact_polygons_pop.py"

    monkeypatch.setattr(starter_mod, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(starter_mod, "load_config", lambda **kwargs: {"paths": {}, "impact_polygons_pop_params": _runtime_params()})

    with pytest.raises(KeyError, match="non_served_nxt_river_outpath"):
        runpy.run_path(str(module_path), run_name="__main__")
