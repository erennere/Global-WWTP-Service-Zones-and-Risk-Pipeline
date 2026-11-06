from __future__ import annotations

import contextlib
import os
import sys
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import duckdb
import pytest
from shapely import from_wkt, to_wkt
from shapely.geometry import box

from wwtp.pop_at_risk_river_calculations import find_diff_pop, find_unserved_pop


pytestmark = pytest.mark.unit


class _DuckResult:
    def __init__(self, dataframe):
        self._dataframe = dataframe

    def df(self):
        return self._dataframe


def _fake_duckdb_connection(conn):
    """Replacement for find_unserved_pop.duckdb_connection yielding ``conn``.

    find_unserved_pop now opens its scratch database through the shared
    ``utils.duckdb_connection`` contextmanager rather than the duckdb module
    global, so tests patch the contextmanager instead of ``duckdb.sql``.
    """

    @contextlib.contextmanager
    def _cm(*args, **kwargs):
        yield conn

    return _cm


class _FakeSqlConnection:
    """Stub conn whose ``sql()`` always returns the same prepared result."""

    def __init__(self, result):
        self._result = result

    def sql(self, query):
        self.query = query
        return self._result


class _FakeDifferenceConnection:
    def __init__(self):
        self.tables = {}

    def execute(self, sql):
        self.sql = sql
        return self

    def register(self, name, dataframe):
        self.tables[name] = dataframe.copy()

    def df(self):
        watershed_df = self.tables["watershed_gdf"]
        pop_df = self.tables["pop_gdf"]
        records = []
        for _, watershed_row in watershed_df.iterrows():
            matches = pop_df[pop_df["HYBAS_ID"] == watershed_row["HYBAS_ID"]]
            if matches.empty:
                continue
            diff_geom = from_wkt(watershed_row["geometry"]).difference(from_wkt(matches.iloc[0]["geometry"]))
            records.append({"HYBAS_ID": watershed_row["HYBAS_ID"], "geometry": to_wkt(diff_geom)})
        return pd.DataFrame(records)

    def close(self):
        return None


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


class _FakeUrbanIntersectionConnection:
    def __init__(self, dataframe):
        self._dataframe = dataframe
        self.tables = {}

    def execute(self, sql):
        self.sql = sql
        return self

    def register(self, name, dataframe):
        self.tables[name] = dataframe.copy()

    def df(self):
        return self._dataframe.copy()

    def close(self):
        return None


def test_find_difference_returns_watershed_minus_served_geometry(monkeypatch, tiny_watershed_gdf):
    served = gpd.GeoDataFrame(
        {"HYBAS_ID": [101], "geometry": [box(-0.08, -0.02, 0.00, 0.02)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: _FakeDifferenceConnection())
    monkeypatch.setattr(find_diff_pop.os.path, "exists", lambda path: False)

    result = find_diff_pop.find_difference(tiny_watershed_gdf.iloc[[0]].copy(), served, basin_col="HYBAS_ID")

    assert result is not None
    assert len(result) == 1
    assert result.iloc[0]["geometry"].intersection(served.iloc[0].geometry).area == pytest.approx(0.0)


def test_find_difference_returns_none_when_duckdb_connect_fails(monkeypatch, tiny_watershed_gdf):
    served = gpd.GeoDataFrame(
        {"HYBAS_ID": [101], "geometry": [box(-0.08, -0.02, 0.00, 0.02)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(
        duckdb,
        "connect",
        lambda path: (_ for _ in ()).throw(RuntimeError("duckdb unavailable")),
    )
    monkeypatch.setattr(find_diff_pop.os.path, "exists", lambda path: False)

    result = find_diff_pop.find_difference(tiny_watershed_gdf.iloc[[0]].copy(), served, basin_col="HYBAS_ID")

    assert result is None


def test_create_unserved_pop_writes_filtered_geometries(monkeypatch, tmp_path):
    input_path = tmp_path / "non_served.csv"
    output_path = tmp_path / "filtered.gpkg"
    urban_path = tmp_path / "urban.gpkg"
    wwtp_path = tmp_path / "wwtp.gpkg"
    captured = {}

    monkeypatch.setattr(find_unserved_pop.os.path, "exists", lambda path: path == str(input_path))
    monkeypatch.setattr(
        find_unserved_pop,
        "duckdb_connection",
        _fake_duckdb_connection(
            _FakeSqlConnection(
                _DuckResult(
                pd.DataFrame(
                    {
                        "pop_sum": ["12"],
                        "geometry": [box(0, 0, 1, 1).wkt],
                    }
                    )
                )
            )
        ),
    )
    monkeypatch.setattr(
        "wwtp.utils.ensure_output_dir_for_file",
        lambda path: captured.setdefault("ensured", path),
    )
    monkeypatch.setattr(
        find_unserved_pop,
        "add_buffers_to_WWTP",
        lambda gdf, **kwargs: captured.update(
            {"wwtp_filepath": kwargs.get("wwtp_filepath"), "wwtp_buffer": kwargs.get("wwtp_buffer")}
        ) or gdf,
    )
    monkeypatch.setattr(
        find_unserved_pop,
        "orchestrate_urban_intersection",
        lambda gdf, urban_filepath, zoom_level=8, max_workers=32, is_parallel=True, urban_buffer=10000, tolerance=0.0001: captured.update(
            {
                "urban_filepath": urban_filepath,
                "zoom_level": zoom_level,
                "max_workers": max_workers,
                "is_parallel": is_parallel,
                "urban_buffer": urban_buffer,
                "tolerance": tolerance,
            }
        ) or gdf,
    )

    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, path, driver=None, index=None, **kwargs):
        captured["path"] = path
        captured["driver"] = driver
        captured["index"] = index
        captured["rows"] = len(self)

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        rows_written = find_unserved_pop.create_unserved_pop(
            str(input_path),
            5,
            str(output_path),
            str(urban_path),
            str(wwtp_path),
            max_workers=7,
            is_parallel=False,
            zoom_level=9,
            urban_buffer=250,
        )
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert rows_written == 1
    assert captured["urban_filepath"] == str(urban_path)
    assert captured["wwtp_filepath"] == str(wwtp_path)
    assert captured["zoom_level"] == 9
    assert captured["max_workers"] == 7
    assert captured["is_parallel"] is False
    assert captured["urban_buffer"] == 250
    assert captured["ensured"] == str(output_path)
    assert captured["path"] == str(output_path)
    assert captured["driver"] == "GPKG"
    assert captured["index"] is False
    assert captured["rows"] == 1


def test_create_unserved_pop_raises_when_input_file_is_missing(monkeypatch, tmp_path):
    input_path = tmp_path / "missing.csv"
    output_path = tmp_path / "filtered.gpkg"
    urban_path = tmp_path / "urban.gpkg"
    wwtp_path = tmp_path / "wwtp.gpkg"

    monkeypatch.setattr(find_unserved_pop.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError, match="Input CSV not found"):
        find_unserved_pop.create_unserved_pop(
            str(input_path), 5, str(output_path), str(urban_path), str(wwtp_path)
        )


def test_create_unserved_pop_writes_empty_output_when_query_returns_no_rows(monkeypatch, tmp_path):
    input_path = tmp_path / "non_served.csv"
    output_path = tmp_path / "filtered.gpkg"
    urban_path = tmp_path / "urban.gpkg"
    wwtp_path = tmp_path / "wwtp.gpkg"
    captured = {}

    monkeypatch.setattr(find_unserved_pop.os.path, "exists", lambda path: path == str(input_path))
    monkeypatch.setattr(
        find_unserved_pop,
        "duckdb_connection",
        _fake_duckdb_connection(_FakeSqlConnection(_DuckResult(pd.DataFrame(columns=["pop_sum", "geometry"])))),
    )
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: captured.setdefault("ensured", path))
    monkeypatch.setattr(find_unserved_pop, "orchestrate_urban_intersection", lambda gdf, *args, **kwargs: gdf)
    monkeypatch.setattr(find_unserved_pop, "add_buffers_to_WWTP", lambda gdf, **kwargs: gdf)

    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, path, driver=None, index=None, **kwargs):
        captured["path"] = path
        captured["rows"] = len(self)
        captured["driver"] = driver
        captured["index"] = index

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        rows_written = find_unserved_pop.create_unserved_pop(
            str(input_path), 5, str(output_path), str(urban_path), str(wwtp_path)
        )
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert rows_written == 0
    assert captured["ensured"] == str(output_path)
    assert captured["path"] == str(output_path)
    assert captured["rows"] == 0
    assert captured["driver"] == "GPKG"
    assert captured["index"] is False


def test_create_unserved_pop_raises_when_geometry_column_is_missing(monkeypatch, tmp_path):
    input_path = tmp_path / "non_served.csv"
    output_path = tmp_path / "filtered.gpkg"
    urban_path = tmp_path / "urban.gpkg"
    wwtp_path = tmp_path / "wwtp.gpkg"

    monkeypatch.setattr(find_unserved_pop.os.path, "exists", lambda path: path == str(input_path))
    monkeypatch.setattr(
        find_unserved_pop,
        "duckdb_connection",
        _fake_duckdb_connection(_FakeSqlConnection(_DuckResult(pd.DataFrame({"pop_sum": ["12"]})))),
    )

    with pytest.raises(ValueError, match="missing required 'geometry' column"):
        find_unserved_pop.create_unserved_pop(
            str(input_path), 5, str(output_path), str(urban_path), str(wwtp_path)
        )


def test_create_unserved_pop_rejects_negative_threshold(monkeypatch, tmp_path):
    input_path = tmp_path / "non_served.csv"
    output_path = tmp_path / "filtered.gpkg"
    urban_path = tmp_path / "urban.gpkg"
    wwtp_path = tmp_path / "wwtp.gpkg"

    monkeypatch.setattr(find_unserved_pop.os.path, "exists", lambda path: path == str(input_path))

    with pytest.raises(ValueError, match="threshold"):
        find_unserved_pop.create_unserved_pop(
            str(input_path), -1, str(output_path), str(urban_path), str(wwtp_path)
        )


def test_find_unserved_pop_main_raises_when_threshold_value_is_missing(mock_cfg, monkeypatch):
    cfg = mock_cfg
    cfg.pop("threshold_value", None)

    monkeypatch.setattr(find_unserved_pop.os, "chdir", lambda path: None)
    monkeypatch.setattr(find_unserved_pop, "parse_args", lambda: SimpleNamespace())
    monkeypatch.setattr(find_unserved_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_unserved_pop, "load_config", lambda **overrides: cfg)

    with pytest.raises(KeyError, match="threshold_value"):
        find_unserved_pop.main()


def test_find_unserved_pop_main_passes_configured_paths_and_threshold(mock_cfg, monkeypatch):
    cfg = mock_cfg
    cfg["threshold_value"] = 37
    cfg["max_workers"] = 11
    cfg["is_parallel"] = False
    cfg["zoom_level"] = 9
    cfg["urban_buffer"] = 600
    cfg["tolerance"] = 0.002
    cfg["wwtp_buffer"] = 750
    cfg["country_output_column"] = "ISO_2"
    cfg["paths"]["non_served_outpath"] = "./outputs/non_served.gpkg"
    cfg["paths"]["non_served_above_threshold_outpath"] = "./outputs/non_served_above_threshold.gpkg"
    cfg["paths"]["urban_areas_filepath"] = "./outputs/urban.gpkg"
    cfg["paths"]["annotated_all_filepath"] = "./outputs/all_annotated.gpkg"
    captured = {}

    monkeypatch.setattr(find_unserved_pop.os, "chdir", lambda path: None)
    monkeypatch.setattr(find_unserved_pop, "parse_args", lambda: SimpleNamespace())
    monkeypatch.setattr(find_unserved_pop, "parse_config_overrides", lambda args=None: {"config": "unused.yaml"})
    monkeypatch.setattr(find_unserved_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(
        find_unserved_pop,
        "create_unserved_pop",
        lambda filepath, threshold, output_filepath, urban_filepath, wwtp_filepath, max_workers=32, is_parallel=True, zoom_level=8, urban_buffer=10000, wwtp_buffer=10000, wwtp_country_col="ISO_2", tolerance=0.0001: captured.update(
            {
                "filepath": filepath,
                "threshold": threshold,
                "output_filepath": output_filepath,
                "urban_filepath": urban_filepath,
                "wwtp_filepath": wwtp_filepath,
                "max_workers": max_workers,
                "is_parallel": is_parallel,
                "zoom_level": zoom_level,
                "urban_buffer": urban_buffer,
                "wwtp_buffer": wwtp_buffer,
                "wwtp_country_col": wwtp_country_col,
                "tolerance": tolerance,
            }
        ) or 4,
    )

    find_unserved_pop.main()

    assert captured == {
        "filepath": os.path.abspath("./outputs/non_served.csv"),
        "threshold": 37,
        "output_filepath": os.path.abspath("./outputs/non_served_above_threshold.gpkg"),
        "urban_filepath": os.path.abspath("./outputs/urban.gpkg"),
        "wwtp_filepath": os.path.abspath("./outputs/all_annotated.gpkg"),
        "max_workers": 11,
        "is_parallel": False,
        "zoom_level": 9,
        "urban_buffer": 600,
        "wwtp_buffer": 750,
        "wwtp_country_col": "ISO_2",
        "tolerance": 0.002,
    }


def test_orchestrate_urban_intersection_filters_matching_unserved_rows(monkeypatch):
    unserved = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 202],
            "geometry": [box(0.0, 0.0, 1.0, 1.0), box(10.0, 10.0, 11.0, 11.0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    urban = gpd.GeoDataFrame(
        {"geometry": [box(0.2, 0.2, 0.8, 0.8)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(find_unserved_pop, "read_urban_areas", lambda filepath: urban.copy())

    def fake_add_bbox(gdf, zoom_level, max_workers, is_parallel):
        gdf["tiles"] = [["0-0-8"] for _ in range(len(gdf))]
        return gdf

    monkeypatch.setattr(find_unserved_pop, "add_bbox", fake_add_bbox)
    monkeypatch.setattr(find_unserved_pop, "ensure_duckdb_spatial", lambda conn: None)
    monkeypatch.setattr(find_unserved_pop, "duckdb_connection", _fake_duckdb_connection(_FakeUrbanIntersectionConnection(
        pd.DataFrame(
            {
                "idx": [0],
                "HYBAS_ID": [101],
                "geometry": [box(0.0, 0.0, 1.0, 1.0).wkt],
                "tile": ["0-0-8"],
                "tiles": [["0-0-8"]],
            }
        )
    )))
    monkeypatch.setattr(find_unserved_pop.os.path, "exists", lambda path: False)

    result = find_unserved_pop.orchestrate_urban_intersection(
        unserved.copy(),
        "urban.gpkg",
        zoom_level=8,
        max_workers=2,
        is_parallel=True,
        urban_buffer=0,
    )

    assert result["HYBAS_ID"].tolist() == [101]
    assert "tiles" not in result.columns


def test_find_differences_serial_mode_combines_multiple_epsg_results(monkeypatch, tiny_watershed_gdf):
    pop_gdf = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 202],
            "geometry": [box(-0.08, -0.08, -0.02, -0.02), box(0.98, -0.02, 1.04, 0.02)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    calls = []

    monkeypatch.setattr(
        find_diff_pop,
        "estimate_utm_epsg_for_geom",
        lambda geom: 32631 if geom.centroid.x < 0.5 else 32632,
    )

    def fake_process_epsg_group(epsg, watershed_gdf, pop_local, basin_col="HYBAS_ID"):
        calls.append((epsg, sorted(pop_local[pop_local["epsg"] == epsg]["HYBAS_ID"].tolist()), basin_col))
        return gpd.GeoDataFrame(
            {"HYBAS_ID": [epsg], "geometry": [box(0.0, 0.0, 1.0, 1.0)]},
            geometry="geometry",
            crs="EPSG:4326",
        )

    monkeypatch.setattr(find_diff_pop, "process_epsg_group", fake_process_epsg_group)

    result = find_diff_pop.find_differences(
        tiny_watershed_gdf,
        pop_gdf,
        is_parallel=False,
        basin_col="HYBAS_ID",
    )

    assert sorted(result["HYBAS_ID"].tolist()) == [32631, 32632]
    assert calls == [
        (32631, [101], "HYBAS_ID"),
        (32632, [202], "HYBAS_ID"),
    ]


def test_find_differences_returns_empty_geodataframe_when_all_groups_skip(monkeypatch, tiny_watershed_gdf):
    pop_gdf = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101],
            "geometry": [box(-0.08, -0.08, -0.02, -0.02)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(find_diff_pop, "estimate_utm_epsg_for_geom", lambda geom: 32631)
    monkeypatch.setattr(find_diff_pop, "process_epsg_group", lambda *args, **kwargs: None)

    result = find_diff_pop.find_differences(
        tiny_watershed_gdf,
        pop_gdf,
        is_parallel=False,
        basin_col="HYBAS_ID",
    )

    assert result.empty
    assert result.crs.to_epsg() == 4326
    assert result.columns.tolist() == tiny_watershed_gdf.columns.tolist()


def test_find_differences_parallel_mode_collects_executor_results(monkeypatch, tiny_watershed_gdf):
    rng = np.random.default_rng(41)
    pop_gdf = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 202],
            "geometry": [
                box(*(rng.random(4) * np.array([0.01, 0.01, 0.02, 0.02]))),
                box(*(np.array([1.0, 0.0, 1.05, 0.05]) + rng.random(4) * 0.01)),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    captured = []

    monkeypatch.setattr(find_diff_pop, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(find_diff_pop, "as_completed", lambda futures: futures)
    monkeypatch.setattr(find_diff_pop, "estimate_utm_epsg_for_geom", lambda geom: 32631 if geom.centroid.x < 0.5 else 32632)

    def fake_process_epsg_group(epsg, watershed_gdf, pop_local, basin_col="HYBAS_ID"):
        captured.append((epsg, basin_col, sorted(pop_local[pop_local["epsg"] == epsg]["HYBAS_ID"].tolist())))
        return gpd.GeoDataFrame(
            {"HYBAS_ID": [epsg], "geometry": [box(0.0, 0.0, 1.0, 1.0)]},
            geometry="geometry",
            crs="EPSG:4326",
        )

    monkeypatch.setattr(find_diff_pop, "process_epsg_group", fake_process_epsg_group)

    result = find_diff_pop.find_differences(
        tiny_watershed_gdf,
        pop_gdf,
        max_workers=2,
        is_parallel=True,
        basin_col="HYBAS_ID",
    )

    assert sorted(result["HYBAS_ID"].tolist()) == [32631, 32632]
    assert captured == [
        (32631, "HYBAS_ID", [101]),
        (32632, "HYBAS_ID", [202]),
    ]


def test_create_unserved_pop_rejects_non_integer_like_threshold(monkeypatch, tmp_path):
    input_path = tmp_path / "non_served.csv"
    output_path = tmp_path / "filtered.gpkg"
    urban_path = tmp_path / "urban.gpkg"

    monkeypatch.setattr(find_unserved_pop.os.path, "exists", lambda path: path == str(input_path))

    with pytest.raises(ValueError, match="integer-like"):
        find_unserved_pop.create_unserved_pop(
            str(input_path), "not-a-number", str(output_path), str(urban_path), str(tmp_path / "wwtp.gpkg")
        )


def test_find_unserved_pop_main_runs_with_module_imports(monkeypatch, tmp_path):
    cfg = {
        "threshold_value": 5,
        "max_workers": 4,
        "is_parallel": True,
        "zoom_level": 8,
        "urban_buffer": 100,
        "tolerance": 0.0001,
        "wwtp_buffer": 10000,
        "country_output_column": "ISO_2",
        "paths": {
            "non_served_outpath": str(tmp_path / "non_served.gpkg"),
            "non_served_above_threshold_outpath": str(tmp_path / "filtered.gpkg"),
            "urban_areas_filepath": str(tmp_path / "urban.gpkg"),
            "annotated_all_filepath": str(tmp_path / "all_annotated.gpkg"),
        },
    }
    captured = {}
    monkeypatch.setattr(find_unserved_pop.os, "chdir", lambda path: None)
    monkeypatch.setattr(find_unserved_pop, "parse_args", lambda: SimpleNamespace())
    monkeypatch.setattr(find_unserved_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_unserved_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(
        find_unserved_pop,
        "create_unserved_pop",
        lambda filepath, threshold, output_filepath, urban_filepath, wwtp_filepath, max_workers=32, is_parallel=True, zoom_level=8, urban_buffer=10000, wwtp_buffer=10000, wwtp_country_col="ISO_2", tolerance=0.0001: captured.update(
            {
                "filepath": filepath,
                "threshold": threshold,
                "output_filepath": output_filepath,
                "urban_filepath": urban_filepath,
                "max_workers": max_workers,
                "is_parallel": is_parallel,
                "zoom_level": zoom_level,
                "urban_buffer": urban_buffer,
            }
        ) or 1,
    )

    find_unserved_pop.main()

    assert captured == {
        "filepath": os.path.abspath(str(tmp_path / "non_served.csv")),
        "threshold": 5,
        "output_filepath": os.path.abspath(str(tmp_path / "filtered.gpkg")),
        "urban_filepath": os.path.abspath(str(tmp_path / "urban.gpkg")),
        "max_workers": 4,
        "is_parallel": True,
        "zoom_level": 8,
        "urban_buffer": 100,
    }


def test_process_epsg_group_returns_none_when_subset_is_empty(tiny_watershed_gdf):
    pop_gdf = gpd.GeoDataFrame(
        {"HYBAS_ID": [101], "epsg": [3857], "geometry": [box(-0.08, -0.08, -0.02, -0.02)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = find_diff_pop.process_epsg_group(
        32631,
        tiny_watershed_gdf,
        pop_gdf,
        basin_col="HYBAS_ID",
    )

    assert result is None


def test_process_epsg_group_reprojects_difference_output_back_to_wgs84(monkeypatch, tiny_watershed_gdf):
    pop_gdf = gpd.GeoDataFrame(
        {"HYBAS_ID": [101], "epsg": [3857], "geometry": [box(-0.08, -0.08, -0.02, -0.02)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    captured = {}

    def fake_find_difference(watershed_gdf, pop_subset, basin_col="HYBAS_ID"):
        captured["watershed_epsg"] = watershed_gdf.crs.to_epsg()
        captured["pop_epsg"] = pop_subset.crs.to_epsg()
        captured["basin_col"] = basin_col
        return pd.DataFrame(
            {"HYBAS_ID": [101], "geometry": [box(-5000.0, -5000.0, -1000.0, -1000.0)]}
        )

    monkeypatch.setattr(find_diff_pop, "find_difference", fake_find_difference)

    result = find_diff_pop.process_epsg_group(
        3857,
        tiny_watershed_gdf,
        pop_gdf,
        basin_col="HYBAS_ID",
    )

    assert result is not None
    assert result.crs.to_epsg() == 4326
    assert captured == {
        "watershed_epsg": 3857,
        "pop_epsg": 3857,
        "basin_col": "HYBAS_ID",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("0", False), ("Yes", True), (False, False)],
)
def test_parse_bool_handles_common_string_values(value, expected):
    assert find_diff_pop.parse_bool(value) is expected


def test_parse_bool_rejects_invalid_value():
    with pytest.raises(ValueError, match="Invalid boolean value"):
        find_diff_pop.parse_bool("not-bool")


def test_parse_args_parses_parallel_flag_and_optional_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "find_diff_pop.py",
            "--is-parallel",
            "false",
            "--level",
            "lvl",
            "--version",
            "v2",
            "--buffer",
            "15000",
            "--weight-method",
            "linear",
            "--weight-func",
            "",
            "--dynamic-buffering",
            "true",
            "--size-measure",
            "footprint",
        ],
    )

    args = find_diff_pop.parse_args()

    assert not hasattr(args, "index")
    assert args.is_parallel is False
    assert args.level == "lvl"
    assert args.version == "v2"
    assert args.buffer == "15000"
    assert args.weight_method == "linear"
    assert args.weight_func == ""
    assert args.dynamic_buffering == "true"
    assert args.size_measure == "footprint"


def test_find_diff_pop_main_reads_the_configured_approach_and_writes_output(
    mock_cfg,
    monkeypatch,
    tiny_watershed_gdf,
):
    cfg = mock_cfg
    cfg["paths"]["watershed"] = "watersheds.gpkg"
    cfg["paths"]["pop_output_dir"] = "./pop_outputs"
    cfg["paths"]["pop_tif_dir"] = "./tifs"
    cfg["paths"]["pop_dif_output_dir"] = "./diff_outputs"
    cfg["max_workers"] = 3
    cfg["basin_column_name"] = "HYBAS_ID"
    cfg["country_output_column"] = "country"
    cfg["figures"] = {"approach": 1}
    captured = {}

    pop_gdf = gpd.GeoDataFrame(
        {"HYBAS_ID": [101], "country": ["DE"], "geometry": [box(-0.08, -0.08, -0.02, -0.02)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    diff_gdf = gpd.GeoDataFrame(
        {"HYBAS_ID": [101], "country": ["DE"], "geometry": [box(-0.08, -0.08, -0.04, -0.04)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(
        find_diff_pop,
        "parse_args",
        lambda: SimpleNamespace(is_parallel=False),
    )
    monkeypatch.setattr(find_diff_pop.os, "chdir", lambda path: None)
    monkeypatch.setattr(find_diff_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_diff_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(find_diff_pop, "create_pop_output_paths", lambda c: {"voronoi": {"1": "/pop_outputs/z_input.gpkg"}})
    monkeypatch.setattr(find_diff_pop.os.path, "exists", lambda path: path == "/pop_outputs/z_input.gpkg")
    monkeypatch.setattr(find_diff_pop.os, "makedirs", lambda path, exist_ok=False: captured.setdefault("makedirs", (path, exist_ok)))

    def fake_read_file(path):
        if path == "/pop_outputs/z_input.gpkg":
            return pop_gdf.copy()
        if path == cfg["paths"]["watershed"]:
            return tiny_watershed_gdf.copy()
        raise AssertionError(f"Unexpected read path: {path}")

    def fake_find_differences(watershed_gdf, pop_subset, max_workers=None, is_parallel=True, basin_col="HYBAS_ID"):
        captured["find_differences"] = {
            "max_workers": max_workers,
            "is_parallel": is_parallel,
            "basin_col": basin_col,
            "pop_rows": len(pop_subset),
            "watershed_rows": len(watershed_gdf),
        }
        return diff_gdf.copy()

    def fake_intersect_all_files(gdf, tif_dir, max_workers=None, country_col=None):
        captured["intersect_all_files"] = {
            "tif_dir": tif_dir,
            "max_workers": max_workers,
            "country_col": country_col,
            "rows": len(gdf),
        }
        return gdf.copy()

    def fake_ensure_output_dir_for_file(path):
        captured["ensured"] = path

    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, path, driver=None, index=None, **kwargs):
        captured["to_file"] = {
            "path": path,
            "driver": driver,
            "index": index,
            "rows": len(self),
        }

    try:
        monkeypatch.setattr(find_diff_pop.gpd, "read_file", fake_read_file)
        monkeypatch.setattr(find_diff_pop, "find_differences", fake_find_differences)
        monkeypatch.setattr(find_diff_pop, "intersect_all_files", fake_intersect_all_files)
        monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", fake_ensure_output_dir_for_file)
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)

        find_diff_pop.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    expected_output = os.path.join(cfg["paths"]["pop_dif_output_dir"], "diff_z_input.gpkg")
    assert captured["makedirs"] == (cfg["paths"]["pop_dif_output_dir"], True)
    assert captured["find_differences"] == {
        "max_workers": 3,
        "is_parallel": False,
        "basin_col": "HYBAS_ID",
        "pop_rows": 1,
        "watershed_rows": 2,
    }
    assert captured["intersect_all_files"] == {
        "tif_dir": cfg["paths"]["pop_tif_dir"],
        "max_workers": 3,
        "country_col": "country",
        "rows": 1,
    }
    assert captured["ensured"] == expected_output
    assert captured["to_file"] == {
        "path": expected_output,
        "driver": "GPKG",
        "index": False,
        "rows": 1,
    }


def test_find_diff_pop_main_raises_when_the_population_layer_is_missing(mock_cfg, monkeypatch):
    cfg = mock_cfg
    cfg["paths"]["watershed"] = "watersheds.gpkg"
    cfg["paths"]["pop_output_dir"] = "./pop_outputs"
    cfg["paths"]["pop_tif_dir"] = "./tifs"
    cfg["paths"]["pop_dif_output_dir"] = "./diff_outputs"
    cfg["figures"] = {"approach": 1}

    monkeypatch.setattr(find_diff_pop, "parse_args", lambda: SimpleNamespace(is_parallel=True))
    monkeypatch.setattr(find_diff_pop.os, "chdir", lambda path: None)
    monkeypatch.setattr(find_diff_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(find_diff_pop, "load_config", lambda **overrides: cfg)

    with pytest.raises(FileNotFoundError, match="run add_pop first"):
        find_diff_pop.main()


