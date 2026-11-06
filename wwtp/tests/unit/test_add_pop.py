from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from wwtp import add_pop


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _add_pop_cfg_defaults(request):
    if "mock_cfg" not in request.fixturenames:
        return

    cfg = request.getfixturevalue("mock_cfg")
    cfg.setdefault("add_pop_max_workers", 8)
    cfg.setdefault("overwrite_existing", False)
    cfg.setdefault("start_year", 2015)
    cfg.setdefault("end_year", 2024)
    cfg.setdefault("include_2014", False)


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


class _RasterStub:
    def __init__(self, crs="EPSG:4326"):
        self.crs = crs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_find_country_tif_files_maps_existing_country_directories(monkeypatch, tmp_path):
    pop_root = tmp_path / "population"
    deu_dir = pop_root / "deu"
    deu_dir.mkdir(parents=True)
    (deu_dir / "worldpop_2020_deu.tif").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        add_pop,
        "get_iso_codes",
        lambda: ({"deu": "DE"}, {"DE": "DEU"}, {}, {}),
    )

    result = add_pop.find_country_tif_files(["DE", "FR", "ZZ"], str(pop_root))

    assert len(result["DE"]) == 1
    assert Path(result["DE"][0]).name == "worldpop_2020_deu.tif"
    assert result["FR"] is None
    assert result["ZZ"] is None


def test_find_country_tif_files_normalizes_whitespace_and_case(monkeypatch, tmp_path):
    pop_root = tmp_path / "population"
    deu_dir = pop_root / "deu"
    deu_dir.mkdir(parents=True)
    (deu_dir / "worldpop_2020_deu.tif").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        add_pop,
        "get_iso_codes",
        lambda: ({"deu": "DE"}, {"DE": "DEU"}, {}, {}),
    )

    result = add_pop.find_country_tif_files([" de ", "de", "DE"], str(pop_root))

    assert list(result.keys()) == ["DE"]
    assert len(result["DE"]) == 1


def test_intersect_single_file_clips_negative_stats_to_zero(monkeypatch, tiny_watershed_gdf):
    tif_path = "synthetic_2020_population.tif"
    original_exists = add_pop.os.path.exists

    monkeypatch.setattr(
        add_pop.os.path,
        "exists",
        lambda path: True if path == tif_path else original_exists(path),
    )
    monkeypatch.setattr(add_pop.rasterio, "open", lambda path: _RasterStub())
    monkeypatch.setattr(
        add_pop,
        "exact_extract",
        lambda rast, vec, ops, output: pd.DataFrame({"sum": [10.0, -5.0], "stdev": [1.0, -2.0]}),
    )

    result = add_pop.intersect_single_file(tiny_watershed_gdf.copy(), [tif_path], all_years=True)

    assert result["2020_zonal_sum"].tolist() == pytest.approx([10.0, 0.0])
    assert result["2020_zonal_std"].tolist() == pytest.approx([1.0, 0.0])


def test_intersect_single_file_drops_empty_geometries_before_extract_and_sets_zero(monkeypatch):
    gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "DE"],
            "geometry": [Polygon([(0, 0), (0, 1), (1, 1), (0, 0)]), Polygon()],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    tif_path = "worldpop_2024_deu.tif"
    monkeypatch.setattr(add_pop.os.path, "exists", lambda path: path == tif_path)
    monkeypatch.setattr(add_pop.rasterio, "open", lambda path: _RasterStub())

    def fake_extract(rast, vec, ops, output):
        assert len(vec) == 1
        return pd.DataFrame({"sum": [12.0], "stdev": [4.0]})

    monkeypatch.setattr(add_pop, "exact_extract", fake_extract)

    result = add_pop.intersect_single_file(gdf.copy(), [tif_path], all_years=True)

    assert result["2024_zonal_sum"].tolist() == pytest.approx([12.0, 0.0])
    assert result["2024_zonal_std"].tolist() == pytest.approx([4.0, 0.0])


def test_find_newest_country_tif_files_selects_highest_year_from_seeded_random_order(monkeypatch, tmp_path):
    rng = np.random.default_rng(7)
    pop_root = tmp_path / "population"
    deu_dir = pop_root / "deu"
    deu_dir.mkdir(parents=True)
    years = rng.permutation([2018, 2020, 2023]).tolist()
    for year in years:
        (deu_dir / f"worldpop_{year}_deu.tif").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        add_pop,
        "get_iso_codes",
        lambda: ({"deu": "DE"}, {"DE": "DEU"}, {}, {}),
    )

    result = add_pop.find_newest_country_tif_files(["DE"], str(pop_root))

    assert Path(result["DE"]).name == "worldpop_2023_deu.tif"


def test_intersect_single_file_returns_original_when_no_usable_year_can_be_parsed(monkeypatch, tiny_watershed_gdf):
    tif_path = "population_unknown_year.tif"
    original_exists = add_pop.os.path.exists

    monkeypatch.setattr(
        add_pop.os.path,
        "exists",
        lambda path: True if path == tif_path else original_exists(path),
    )
    monkeypatch.setattr(
        add_pop.rasterio,
        "open",
        lambda path: (_ for _ in ()).throw(AssertionError("raster should not be opened when no year is parseable")),
    )

    result = add_pop.intersect_single_file(tiny_watershed_gdf.copy(), [tif_path], all_years=True)

    assert result.crs == tiny_watershed_gdf.crs
    assert result.columns.tolist() == tiny_watershed_gdf.columns.tolist()


def test_intersect_all_files_preserves_input_crs_and_concatenates_seeded_random_country_slices(monkeypatch):
    rng = np.random.default_rng(19)
    point_a, point_b = rng.random(2), rng.random(2) + 5
    gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "FR"],
            "value": rng.integers(10, 100, size=2),
            "geometry": [Point(point_a[0], point_a[1]), Point(point_b[0], point_b[1])],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )

    monkeypatch.setattr(add_pop, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(add_pop, "as_completed", lambda futures: futures)
    monkeypatch.setattr(add_pop, "tqdm", lambda iterable, total=None, desc=None: iterable)
    monkeypatch.setattr(add_pop.random, "shuffle", lambda seq: seq.reverse())
    monkeypatch.setattr(
        add_pop,
        "find_country_tif_files",
        lambda countries, tif_dir: {"DE": ["de_2020.tif"], "FR": ["fr_2020.tif"]},
    )

    def fake_intersect_single_file(sub_gdf, tif_paths, all_years=True, years=None):
        return sub_gdf.assign(pop_sum_seed=int(rng.integers(1, 50)))

    monkeypatch.setattr(add_pop, "intersect_single_file", fake_intersect_single_file)

    result = add_pop.intersect_all_files(gdf, tif_dir="unused", max_workers=1, all_years=True, country_col="ISO_2")

    assert result.crs is not None
    assert result.crs.to_epsg() == 3857
    assert set(result["ISO_2"]) == {"DE", "FR"}
    assert "pop_sum_seed" in result.columns


def test_intersect_all_files_keeps_rows_of_countries_without_rasters(monkeypatch):
    """R7: polygons in countries without a raster used to vanish from the output."""
    gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["ZZ"],
            "geometry": [Point(1, 2)],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )

    monkeypatch.setattr(add_pop, "find_country_tif_files", lambda countries, tif_dir: {"ZZ": None})
    monkeypatch.setattr(add_pop, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(add_pop, "as_completed", lambda futures: futures)
    monkeypatch.setattr(add_pop, "tqdm", lambda iterable, total=None, desc=None: iterable)

    result = add_pop.intersect_all_files(gdf, tif_dir="unused", max_workers=1, all_years=True, country_col="ISO_2")

    assert len(result) == 1
    assert result["ISO_2"].tolist() == ["ZZ"]
    assert result.crs is not None
    assert result.crs.to_epsg() == 3857
    assert "geometry" in result.columns


def test_intersect_all_files_keeps_rows_with_invalid_country_and_marks_population_missing(monkeypatch):
    """R7: rows with country XX/empty and rows of countries without rasters stay, with NaN population."""
    gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", None, " ", "ZZ"],
            "site": [1, 2, 3, 4],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(add_pop, "find_country_tif_files", lambda countries, tif_dir: {"DE": ["de_2020.tif"], "ZZ": None})
    monkeypatch.setattr(add_pop, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(add_pop, "as_completed", lambda futures: futures)
    monkeypatch.setattr(add_pop, "tqdm", lambda iterable, total=None, desc=None: iterable)
    monkeypatch.setattr(
        add_pop, "intersect_single_file", lambda sub, paths, all_years=True, years=None: sub.assign(**{"2020_zonal_sum": 50.0})
    )

    result = add_pop.intersect_all_files(gdf, tif_dir="unused", max_workers=1, country_col="ISO_2")

    assert sorted(result["site"]) == [1, 2, 3, 4]
    by_site = result.set_index("site")["2020_zonal_sum"]
    assert by_site[1] == 50.0
    assert by_site[[2, 3, 4]].isna().all()


def test_intersect_all_files_rejects_non_positive_workers(monkeypatch):
    gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE"],
            "geometry": [Point(1, 2)],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="max_workers"):
        add_pop.intersect_all_files(gdf, tif_dir="unused", max_workers=0, all_years=True, country_col="ISO_2")


def test_orchestrate_intersections_rejects_negative_index(tmp_path):
    data_dir = tmp_path / "voronoi"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    (data_dir / "sample.gpkg").write_text("stub", encoding="utf-8")

    with pytest.raises(IndexError, match="out of range"):
        add_pop.orchestrate_intersections(str(data_dir), "unused", str(output_dir), index=-1)


def test_orchestrate_intersections_rejects_index_past_available_files(tmp_path):
    data_dir = tmp_path / "voronoi"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    (data_dir / "sample.gpkg").write_text("stub", encoding="utf-8")

    with pytest.raises(IndexError, match="out of range"):
        add_pop.orchestrate_intersections(str(data_dir), "unused", str(output_dir), index=1)


def test_orchestrate_intersections_reads_selected_file_and_writes_output(monkeypatch, tmp_path):
    data_dir = tmp_path / "voronoi"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    (data_dir / "b_file.gpkg").write_text("stub", encoding="utf-8")
    (data_dir / "a_file.gpkg").write_text("stub", encoding="utf-8")
    rng = np.random.default_rng(31)
    captured = {}

    source_gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "FR"],
            "geometry": [Point(*rng.random(2)), Point(*(rng.random(2) + 1))],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    def fake_read_file(path):
        captured["read_path"] = path
        return source_gdf.copy()

    def fake_intersect_all_files(gdf, tif_dir, max_workers, all_years=True, country_col="ISO_2", years=None):
        captured["intersect"] = {
            "tif_dir": tif_dir,
            "max_workers": max_workers,
            "country_col": country_col,
            "rows": len(gdf),
        }
        return gdf.assign(seed_value=int(rng.integers(1, 1000)))

    monkeypatch.setattr(add_pop.gpd, "read_file", fake_read_file)
    monkeypatch.setattr(add_pop, "intersect_all_files", fake_intersect_all_files)
    monkeypatch.setattr(add_pop, "ensure_output_dir_for_file", lambda path: captured.setdefault("ensured", path))

    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, path, driver=None, index=None, **kwargs):
        captured["write"] = {"path": path, "driver": driver, "index": index, "rows": len(self)}

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        add_pop.orchestrate_intersections(str(data_dir), "tifs", str(output_dir), index=1, max_workers=3, country_col="ISO_2")
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    expected_input = str(data_dir / "b_file.gpkg")
    expected_output = str(output_dir / "pop_added_b_file.gpkg")
    assert captured["read_path"] == expected_input
    assert captured["intersect"] == {"tif_dir": "tifs", "max_workers": 3, "country_col": "ISO_2", "rows": 2}
    assert captured["ensured"] == expected_output
    assert captured["write"] == {"path": expected_output, "driver": "GPKG", "index": False, "rows": 2}


def test_main_passes_configured_paths_and_avoids_false_existing_output_warning(monkeypatch, mock_cfg, caplog, tmp_path):
    cfg = mock_cfg
    cfg["paths"]["voronoi_dir"] = str(tmp_path / "voronoi")
    cfg["paths"]["pop_tif_dir"] = str(tmp_path / "tifs")
    cfg["paths"]["pop_output_dir"] = str(tmp_path / "new_output")
    cfg["country_output_column"] = "ISO_2"
    cfg["add_pop_max_workers"] = 5
    cfg["overwrite_existing"] = False
    captured = {}
    state = {"output_exists": False}

    monkeypatch.setattr(sys, "argv", ["add_pop.py", "--index", "2"])
    monkeypatch.setattr(add_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(add_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(add_pop.os, "chdir", lambda path: None)

    def fake_makedirs(path, exist_ok=False):
        captured.setdefault("makedirs", (path, exist_ok))
        if path == cfg["paths"]["pop_output_dir"]:
            state["output_exists"] = True

    monkeypatch.setattr(add_pop.os, "makedirs", fake_makedirs)

    original_exists = add_pop.os.path.exists
    monkeypatch.setattr(
        add_pop.os.path,
        "exists",
        lambda path: state["output_exists"] if path == cfg["paths"]["pop_output_dir"] else original_exists(path),
    )
    monkeypatch.setattr(
        add_pop,
        "orchestrate_intersections",
        lambda data_dir, tif_dir, output_dir, index, max_workers, country_col=None, overwrite=True, years=None: captured.update(
            {
                "call": {
                    "data_dir": data_dir,
                    "tif_dir": tif_dir,
                    "output_dir": output_dir,
                    "index": index,
                    "max_workers": max_workers,
                    "country_col": country_col,
                    "overwrite": overwrite,
                }
            }
        ),
    )

    with caplog.at_level(logging.WARNING):
        add_pop.main()

    assert captured["makedirs"] == (cfg["paths"]["pop_output_dir"], True)
    assert captured["call"] == {
        "data_dir": cfg["paths"]["voronoi_dir"],
        "tif_dir": cfg["paths"]["pop_tif_dir"],
        "output_dir": cfg["paths"]["pop_output_dir"],
        "index": 2,
        "max_workers": 5,
        "country_col": "ISO_2",
        "overwrite": False,
    }


def test_main_exits_on_non_integer_index(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["add_pop.py", "--index", "not-an-int"])

    with pytest.raises(SystemExit, match="2"):
        add_pop.main()


def test_main_exits_when_index_argument_is_missing(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["add_pop.py"])

    with pytest.raises(SystemExit, match="2"):
        add_pop.main()


def test_main_exits_when_required_path_key_missing(monkeypatch, mock_cfg):
    cfg = mock_cfg
    cfg["paths"].pop("voronoi_dir", None)

    monkeypatch.setattr(sys, "argv", ["add_pop.py", "--index", "0"])
    monkeypatch.setattr(add_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(add_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(add_pop.os, "chdir", lambda path: None)

    with pytest.raises(KeyError, match="voronoi_dir"):
        add_pop.main()


def test_main_exits_when_paths_section_is_empty(monkeypatch, mock_cfg):
    cfg = mock_cfg
    cfg["paths"] = {}

    monkeypatch.setattr(sys, "argv", ["add_pop.py", "--index", "0"])
    monkeypatch.setattr(add_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(add_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(add_pop.os, "chdir", lambda path: None)

    with pytest.raises(KeyError, match="voronoi_dir"):
        add_pop.main()


def test_orchestrate_intersections_skips_when_output_exists_and_overwrite_disabled(monkeypatch, caplog, tmp_path):
    data_dir = tmp_path / "voronoi"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    output_dir.mkdir()
    (data_dir / "b_file.gpkg").write_text("stub", encoding="utf-8")
    (output_dir / "pop_added_b_file.gpkg").write_text("stub", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(add_pop, "intersect_all_files", lambda *a, **k: captured.setdefault("processed", True))

    with caplog.at_level(logging.INFO):
        add_pop.orchestrate_intersections(str(data_dir), "tifs", str(output_dir), index=0, overwrite=False)

    assert "processed" not in captured
    assert "already exists" in caplog.text and "overwrite is False" in caplog.text


def test_orchestrate_intersections_reprocesses_when_overwrite_enabled(monkeypatch, tmp_path):
    data_dir = tmp_path / "voronoi"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    output_dir.mkdir()
    (data_dir / "b_file.gpkg").write_text("stub", encoding="utf-8")
    (output_dir / "pop_added_b_file.gpkg").write_text("stale", encoding="utf-8")
    captured = {}

    source_gdf = gpd.GeoDataFrame(
        {"ISO_2": ["DE"], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    monkeypatch.setattr(add_pop.gpd, "read_file", lambda path: source_gdf.copy())
    def _fake_intersect(gdf, *a, **k):
        captured["processed"] = True
        return gdf

    monkeypatch.setattr(add_pop, "intersect_all_files", _fake_intersect)
    monkeypatch.setattr(add_pop, "ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, *a, **k: None)

    add_pop.orchestrate_intersections(str(data_dir), "tifs", str(output_dir), index=0, overwrite=True)

    assert captured.get("processed") is True


def test_main_exits_when_config_overrides_are_invalid(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["add_pop.py", "--index", "0"])
    monkeypatch.setattr(add_pop, "parse_config_overrides", lambda args=None: (_ for _ in ()).throw(ValueError("invalid override")))

    with pytest.raises(SystemExit, match="1"):
        add_pop.main()


def test_intersect_single_file_returns_empty_input_unchanged(tiny_watershed_gdf):
    empty = tiny_watershed_gdf.iloc[0:0].copy()

    result = add_pop.intersect_single_file(empty, ["unused_2020.tif"], all_years=True)

    assert result.empty
    assert result.columns.tolist() == empty.columns.tolist()


def test_intersect_single_file_raises_with_context_on_extract_failure(monkeypatch):
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:3857",
    )
    existing = {"country_2020_deu.tif", "country_2021_deu.tif"}

    monkeypatch.setattr(add_pop.os.path, "exists", lambda path: path in existing)
    monkeypatch.setattr(add_pop.rasterio, "open", lambda path: _RasterStub("EPSG:4326"))

    def fake_exact_extract(rast, vec, ops, output):
        if rast == "country_2020_deu.tif":
            raise RuntimeError("bad extract")
        return pd.DataFrame({"sum": [5.0], "stdev": [2.0]})

    monkeypatch.setattr(add_pop, "exact_extract", fake_exact_extract)

    with pytest.raises(RuntimeError, match="country=|country_2020_deu"):
        add_pop.intersect_single_file(
            gdf.copy(),
            ["missing_2019_deu.tif", "country_2020_deu.tif", "country_2021_deu.tif"],
            all_years=True,
        )


def test_intersect_single_file_latest_only_skips_older_years(monkeypatch, tiny_watershed_gdf):
    calls = []

    monkeypatch.setattr(add_pop.os.path, "exists", lambda path: True)
    monkeypatch.setattr(add_pop.rasterio, "open", lambda path: _RasterStub())
    monkeypatch.setattr(
        add_pop,
        "exact_extract",
        lambda rast, vec, ops, output: calls.append(rast) or pd.DataFrame({"sum": [3.0, 4.0], "stdev": [1.0, 1.5]}),
    )

    result = add_pop.intersect_single_file(
        tiny_watershed_gdf.copy(),
        ["country_2020_deu.tif", "country_2022_deu.tif"],
        all_years=False,
    )

    assert calls == ["country_2022_deu.tif"]
    assert "2020_zonal_sum" not in result.columns
    assert "2022_zonal_sum" in result.columns


def test_find_country_tif_files_returns_none_for_known_country_without_directory(monkeypatch, tmp_path):
    pop_root = tmp_path / "population"
    pop_root.mkdir()

    monkeypatch.setattr(
        add_pop,
        "get_iso_codes",
        lambda: ({"deu": "DE"}, {"DE": "DEU"}, {}, {}),
    )

    result = add_pop.find_country_tif_files(["DE"], str(pop_root))

    assert result == {"DE": None}


def test_find_newest_country_tif_files_skips_missing_lists_and_unparseable_names(monkeypatch, caplog):
    monkeypatch.setattr(
        add_pop,
        "find_country_tif_files",
        lambda countries, tif_dir: {"DE": None, "FR": ["fr_population_latest.tif"]},
    )
    monkeypatch.setattr(add_pop.os.path, "exists", lambda path: True)

    with caplog.at_level(logging.WARNING):
        result = add_pop.find_newest_country_tif_files(["DE", "FR"], "unused")

    assert result == {}
    assert "Could not parse year from filename" in caplog.text


def test_intersect_all_files_raises_on_future_result_errors(monkeypatch):
    gdf = gpd.GeoDataFrame(
        {"ISO_2": ["DE"], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    class _ErrorFuture:
        def result(self):
            raise RuntimeError("future failed")

    class _ErrorExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return _ErrorFuture()

    monkeypatch.setattr(add_pop, "ProcessPoolExecutor", _ErrorExecutor)
    monkeypatch.setattr(add_pop, "as_completed", lambda futures: futures)
    monkeypatch.setattr(add_pop, "tqdm", lambda iterable, total=None, desc=None: iterable)
    monkeypatch.setattr(add_pop.random, "shuffle", lambda seq: None)
    monkeypatch.setattr(add_pop, "find_country_tif_files", lambda countries, tif_dir: {"DE": ["de_2020.tif"]})

    with pytest.raises(RuntimeError, match="failed for"):
        add_pop.intersect_all_files(gdf, tif_dir="unused", max_workers=1, all_years=True, country_col="ISO_2")


def test_orchestrate_intersections_reraises_read_errors(monkeypatch, tmp_path):
    data_dir = tmp_path / "voronoi"
    data_dir.mkdir()
    (data_dir / "sample.gpkg").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(add_pop.gpd, "read_file", lambda path: (_ for _ in ()).throw(RuntimeError("broken layer")))

    with pytest.raises(RuntimeError, match="broken layer"):
        add_pop.orchestrate_intersections(str(data_dir), "unused", str(tmp_path / "output"), index=0)


def test_main_exits_when_orchestration_fails(monkeypatch, mock_cfg, tmp_path):
    cfg = mock_cfg
    cfg["paths"]["voronoi_dir"] = str(tmp_path / "voronoi")
    cfg["paths"]["pop_tif_dir"] = str(tmp_path / "tifs")
    cfg["paths"]["pop_output_dir"] = str(tmp_path / "output")
    cfg["country_output_column"] = "ISO_2"
    cfg["add_pop_max_workers"] = 2

    monkeypatch.setattr(sys, "argv", ["add_pop.py", "--index", "0"])
    monkeypatch.setattr(add_pop, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(add_pop, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(add_pop.os, "chdir", lambda path: None)
    monkeypatch.setattr(add_pop.os.path, "exists", lambda path: False)
    monkeypatch.setattr(add_pop.os, "makedirs", lambda path, exist_ok=False: None)
    monkeypatch.setattr(
        add_pop,
        "orchestrate_intersections",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("orchestration failed")),
    )

    with pytest.raises(SystemExit, match="1"):
        add_pop.main()





def test_intersect_single_file_repairs_invalid_polygons_and_leaves_null_geometry_unmeasured(monkeypatch):
    """R7: an invalid polygon used to be dropped and reported as 0 people."""
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])  # self-intersecting
    gdf = gpd.GeoDataFrame(
        {"ISO_2": ["DE", "DE", "DE"], "geometry": [bowtie, None, Polygon()]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    tif_path = "worldpop_2024_deu.tif"
    monkeypatch.setattr(add_pop.os.path, "exists", lambda path: path == tif_path)
    monkeypatch.setattr(add_pop.rasterio, "open", lambda path: _RasterStub())

    def fake_extract(rast, vec, ops, output):
        assert len(vec) == 1 and vec.geometry.iloc[0].is_valid
        return pd.DataFrame({"sum": [7.0], "stdev": [1.0]})

    monkeypatch.setattr(add_pop, "exact_extract", fake_extract)

    result = add_pop.intersect_single_file(gdf.copy(), [tif_path])

    sums = result["2024_zonal_sum"].tolist()
    assert sums[0] == 7.0
    assert np.isnan(sums[1])
    assert sums[2] == 0.0



def test_population_years_follow_download_pop(tmp_path):
    """A 2014 raster left from an older download is not read unless include_2014 is on."""
    assert add_pop.population_years({"start_year": 2015, "end_year": 2024, "include_2014": False}) == set(range(2015, 2025))
    assert 2014 in add_pop.population_years({"start_year": 2015, "end_year": 2024, "include_2014": True})


def test_rasters_of_unconfigured_years_are_skipped(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    paths = []
    for year in (2014, 2020):
        path = tmp_path / f"XX_pop_{year}_CN_100m.tif"
        with rasterio.open(path, "w", driver="GTiff", height=2, width=2, count=1, dtype="float32", crs="EPSG:4326",
                           transform=from_origin(8.0, 50.0, 0.01, 0.01)) as dst:
            dst.write(np.ones((2, 2), dtype="float32"), 1)
        paths.append(str(path))
    gdf = gpd.GeoDataFrame({"ISO_2": ["XX"]}, geometry=[box(8.0, 49.98, 8.02, 50.0)], crs=4326)
    out = add_pop.intersect_single_file(gdf, paths, years={2020})
    assert "2020_zonal_sum" in out.columns and "2014_zonal_sum" not in out.columns
