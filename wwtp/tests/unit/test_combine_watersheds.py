"""Tests for combine_watersheds (moved from the coverage-campaign files)."""
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon
from wwtp import combine_watersheds
from wwtp.pop_at_risk_river_calculations import assign_rivers_to_basin

pytestmark = pytest.mark.unit


def test_extract_and_merge_geodata_merges_first_readable_layer_per_zip(monkeypatch, tmp_path):
    zip_dir = tmp_path / "zips"
    out_dir = tmp_path / "out"
    zip_dir.mkdir()

    (zip_dir / "a.zip").write_bytes(b"PK\x03\x04fake")
    (zip_dir / "b.zip").write_bytes(b"PK\x03\x04fake")

    captured = {}

    def fake_read_file(path):
        name = Path(path).name
        if name in {"candidate_a.geojson", "candidate_b.geojson"}:
            return gpd.GeoDataFrame(
                {"source": [name], "geometry": [Point(0, 0)]},
                geometry="geometry",
                crs="EPSG:4326",
            )
        raise RuntimeError("not geospatial")

    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, path, driver=None, index=None, **kwargs):
        captured["rows"] = len(self)
        captured["path"] = path
        captured["driver"] = driver

    tmp_roots = []

    class _FakeTmpDir:
        def __enter__(self):
            root = tmp_path / f"tmp_{len(tmp_roots)}"
            root.mkdir(exist_ok=True)
            tmp_roots.append(root)
            return str(root)

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeZip:
        def __init__(self, zip_path, mode="r"):
            self.zip_path = Path(zip_path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extractall(self, target):
            root = Path(target)
            if self.zip_path.name == "a.zip":
                (root / "candidate_a.geojson").write_text("{}", encoding="utf-8")
                (root / "skip.txt").write_text("x", encoding="utf-8")
            elif self.zip_path.name == "b.zip":
                (root / "candidate_b.geojson").write_text("{}", encoding="utf-8")

    def fake_walk(root):
        root = Path(root)
        files = sorted(p.name for p in root.iterdir() if p.is_file())
        yield str(root), [], files

    monkeypatch.setattr(combine_watersheds.tempfile, "TemporaryDirectory", _FakeTmpDir)
    monkeypatch.setattr(combine_watersheds.zipfile, "ZipFile", _FakeZip)
    monkeypatch.setattr(combine_watersheds.os, "walk", fake_walk)
    monkeypatch.setattr(combine_watersheds.gpd, "read_file", fake_read_file)
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: captured.setdefault("ensured", str(path)))

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        combine_watersheds.extract_and_merge_geodata(zip_dir, out_dir, output_filename="merged.gpkg")
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert captured["ensured"].endswith("merged.gpkg")
    assert captured["rows"] == 2
    assert captured["driver"] == "GPKG"


def test_extract_and_merge_geodata_propagates_bad_zip(monkeypatch, tmp_path):
    """A corrupt archive must abort the merge, not be skipped silently."""
    zip_dir = tmp_path / "zips"
    zip_dir.mkdir()
    (zip_dir / "bad.zip").write_text("not-a-zip", encoding="utf-8")

    class _FakeTmpDir:
        def __enter__(self):
            root = tmp_path / "tmp"
            root.mkdir(exist_ok=True)
            return str(root)

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeZip:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise combine_watersheds.zipfile.BadZipFile("bad")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(combine_watersheds.tempfile, "TemporaryDirectory", _FakeTmpDir)
    monkeypatch.setattr(combine_watersheds.zipfile, "ZipFile", _FakeZip)

    with pytest.raises(combine_watersheds.zipfile.BadZipFile):
        combine_watersheds.extract_and_merge_geodata(zip_dir, tmp_path / "out")


def test_extract_and_merge_geodata_handles_no_readable_layers(monkeypatch, tmp_path):
    zip_dir = tmp_path / "zips"
    out_dir = tmp_path / "out"
    zip_dir.mkdir()
    (zip_dir / "a.zip").write_bytes(b"PK\x03\x04fake")

    class _FakeTmpDir:
        def __enter__(self):
            root = tmp_path / "tmp"
            root.mkdir(exist_ok=True)
            return str(root)

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeZip:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extractall(self, target):
            (Path(target) / "notes.txt").write_text("x", encoding="utf-8")

    monkeypatch.setattr(combine_watersheds.tempfile, "TemporaryDirectory", _FakeTmpDir)
    monkeypatch.setattr(combine_watersheds.zipfile, "ZipFile", _FakeZip)
    monkeypatch.setattr(combine_watersheds.os, "walk", lambda root: [(str(root), [], ["notes.txt"])])
    monkeypatch.setattr(combine_watersheds.gpd, "read_file", lambda path: (_ for _ in ()).throw(RuntimeError("bad")))

    called = {"to_file": False}
    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, *args, **kwargs):
        called["to_file"] = True

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        with pytest.raises(RuntimeError, match="No readable geospatial layer"):
            combine_watersheds.extract_and_merge_geodata(zip_dir, out_dir)
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert called["to_file"] is False


def test_extract_and_merge_geodata_reads_only_one_layer_per_zip_across_subdirs(monkeypatch, tmp_path):
    zip_dir = tmp_path / "zips"
    out_dir = tmp_path / "out"
    zip_dir.mkdir()
    (zip_dir / "a.zip").write_bytes(b"PK\x03\x04fake")

    captured = {}

    class _FakeTmpDir:
        def __enter__(self):
            root = tmp_path / "tmp_nested"
            (root / "d1").mkdir(parents=True, exist_ok=True)
            (root / "d2").mkdir(parents=True, exist_ok=True)
            return str(root)

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeZip:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extractall(self, target):
            root = Path(target)
            (root / "d1" / "candidate_1.geojson").write_text("{}", encoding="utf-8")
            (root / "d2" / "candidate_2.geojson").write_text("{}", encoding="utf-8")

    def fake_walk(root):
        root = Path(root)
        yield str(root / "d1"), [], ["candidate_1.geojson"]
        yield str(root / "d2"), [], ["candidate_2.geojson"]

    def fake_read_file(path):
        return gpd.GeoDataFrame(
            {"source": [Path(path).name], "geometry": [Point(0, 0)]},
            geometry="geometry",
            crs="EPSG:4326",
        )

    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, path, driver=None, index=None, **kwargs):
        captured["rows"] = len(self)

    monkeypatch.setattr(combine_watersheds.tempfile, "TemporaryDirectory", _FakeTmpDir)
    monkeypatch.setattr(combine_watersheds.zipfile, "ZipFile", _FakeZip)
    monkeypatch.setattr(combine_watersheds.os, "walk", fake_walk)
    monkeypatch.setattr(combine_watersheds.gpd, "read_file", fake_read_file)
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: None)

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        combine_watersheds.extract_and_merge_geodata(zip_dir, out_dir, output_filename="merged.gpkg")
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert captured["rows"] == 1


def test_combine_watersheds_main_handles_missing_optional_overrides(monkeypatch, tmp_path):
    base = tmp_path / "levels"
    (base / "lvl1").mkdir(parents=True)
    (base / "misc").mkdir(parents=True)

    cfg_initial = {
        "paths": {
            "watersheds_zip_dir": str(base / "lvl1"),
            "watershed": str(tmp_path / "out" / "lvl1.gpkg"),
        }
    }

    calls = []

    def fake_load_config(**kwargs):
        if kwargs:
            level = kwargs.get("level")
            return {
                "paths": {
                    "watersheds_zip_dir": str(base / f"lvl{level}"),
                    "watershed": str(tmp_path / "out" / f"lvl{level}.gpkg"),
                }
            }
        return cfg_initial

    monkeypatch.setattr(
        combine_watersheds,
        "parse_config_overrides",
        lambda *a, **k: {
            "level": None,
            "version": None,
            "buffer": None,
            "weight_method": None,
            "weight_func": None,
            "dynamic_buffering": None,
            "size_measure": None,
        },
    )
    monkeypatch.setattr(combine_watersheds, "load_config", fake_load_config)
    monkeypatch.setattr(combine_watersheds.os, "listdir", lambda p: ["lvl1", "misc"])
    monkeypatch.setattr(combine_watersheds.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(
        combine_watersheds,
        "extract_and_merge_geodata",
        lambda zip_dir, out_dir, output_filename="merged.gpkg": calls.append((zip_dir, out_dir, output_filename)),
    )

    combine_watersheds.main()

    assert len(calls) == 1
    assert calls[0][0].endswith("lvl1")
    assert calls[0][2] == "lvl1.gpkg"


def test_combine_watersheds_main_requires_levels_root(monkeypatch, tmp_path):
    cfg_initial = {
        "paths": {
            "watersheds_zip_dir": str(tmp_path / "missing" / "lvl1"),
            "watershed": str(tmp_path / "out" / "lvl1.gpkg"),
        }
    }

    monkeypatch.setattr(combine_watersheds.os, "chdir", lambda path: None)
    monkeypatch.setattr(combine_watersheds, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(combine_watersheds, "load_config", lambda **kwargs: cfg_initial)
    monkeypatch.setattr(combine_watersheds.os.path, "isdir", lambda p: False)

    with pytest.raises(FileNotFoundError, match="levels root"):
        combine_watersheds.main()


def test_combine_watersheds_main_honors_explicit_level_override(monkeypatch, tmp_path):
    """An explicit --level processes only that level (combine_watersheds F1).

    The flag used to be parsed and then discarded in favour of discovering every
    ``lvl*`` directory. This pins the honor-the-flag branch, and the companion
    test below pins that omitting the flag still discovers every level.
    """
    calls = []
    listdir_calls = []

    def fake_load_config(**kwargs):
        level = kwargs.get("level")
        return {
            "paths": {
                "watersheds_zip_dir": str(tmp_path / f"lvl{level}"),
                "watershed": str(tmp_path / "out" / f"lvl{level}.gpkg"),
            }
        }

    monkeypatch.setattr(combine_watersheds.os, "chdir", lambda path: None)
    monkeypatch.setattr(
        combine_watersheds,
        "parse_config_overrides",
        lambda *a, **k: {"level": "9", "version": None, "buffer": None},
    )
    monkeypatch.setattr(combine_watersheds, "load_config", fake_load_config)
    monkeypatch.setattr(combine_watersheds.os, "listdir", lambda p: listdir_calls.append(p) or [])
    monkeypatch.setattr(
        combine_watersheds,
        "extract_and_merge_geodata",
        lambda zip_dir, out_dir, output_filename="merged.gpkg": calls.append((zip_dir, output_filename)),
    )

    combine_watersheds.main()

    assert len(calls) == 1
    assert calls[0][0].endswith("lvl9")
    assert calls[0][1] == "lvl9.gpkg"
    # The discovery branch must be skipped entirely when --level is given.
    assert listdir_calls == []


def test_combine_watersheds_main_discovers_every_level_without_override(monkeypatch, tmp_path):
    """No --level keeps the original discover-all-levels behavior."""
    calls = []

    def fake_load_config(**kwargs):
        level = kwargs.get("level")
        if level is None:
            return {
                "paths": {
                    "watersheds_zip_dir": str(tmp_path / "levels" / "lvl6"),
                    "watershed": str(tmp_path / "out" / "lvl6.gpkg"),
                }
            }
        return {
            "paths": {
                "watersheds_zip_dir": str(tmp_path / "levels" / f"lvl{level}"),
                "watershed": str(tmp_path / "out" / f"lvl{level}.gpkg"),
            }
        }

    monkeypatch.setattr(combine_watersheds.os, "chdir", lambda path: None)
    monkeypatch.setattr(combine_watersheds, "parse_config_overrides", lambda *a, **k: {"level": None})
    monkeypatch.setattr(combine_watersheds, "load_config", fake_load_config)
    monkeypatch.setattr(combine_watersheds.os, "listdir", lambda p: ["lvl6", "lvl9", "notalevel"])
    monkeypatch.setattr(combine_watersheds.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(
        combine_watersheds,
        "extract_and_merge_geodata",
        lambda zip_dir, out_dir, output_filename="merged.gpkg": calls.append(output_filename),
    )

    combine_watersheds.main()

    assert sorted(calls) == ["lvl6.gpkg", "lvl9.gpkg"]
