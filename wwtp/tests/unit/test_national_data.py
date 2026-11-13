"""wwtp.pop_validation_scripts.national_data: downloads, file descriptions and the new register plants."""

import json
import os
import zipfile

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from wwtp.pop_validation_scripts import national_data as nd

pytestmark = pytest.mark.unit


class Response:
    def __init__(self, url, payload=None, content=b"", fail=False):
        self.url, self._payload, self._content, self._fail = url, payload, content, fail
        self.headers = {"content-type": "application/test"}

    def raise_for_status(self):
        if self._fail:
            raise OSError("403 Forbidden")

    def json(self):
        return self._payload

    def iter_content(self, chunk_size):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _feature(i):
    return {"type": "Feature", "properties": {"id": i}, "geometry": {"type": "Point", "coordinates": [i, 0]}}


class Session:
    """http: bytes; ogcapi: two pages joined by a next link; arcgis: pages by resultOffset."""

    def __init__(self):
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None, stream=False):
        self.calls.append((url, params))
        if "broken" in url:
            return Response(url, fail=True)
        if url.endswith("/items"):
            return Response(
                url, {"features": [_feature(0), _feature(1)], "links": [{"rel": "next", "href": url + "?page=2"}]}
            )
        if url.endswith("?page=2"):
            return Response(url, {"features": [_feature(2)], "links": [{"rel": "self", "href": url}]})
        if url.endswith("/query"):
            start = params["resultOffset"]
            return Response(
                url, {"features": [_feature(i) for i in range(start, min(start + params["resultRecordCount"], 5))]}
            )
        return Response(url + "#final", content=b"col\n1\n")


HW_COLUMNS = {"id": "WASTE_ID", "name": "WWTP_NAME", "country": "CNTRY_ISO", "lon": "LON_WWTP", "lat": "LAT_WWTP"}


def _cfg(tmp_path, sources, registers=None):
    pd.DataFrame(  # HydroWASTE: one Australian record 300 m from the register plant, one Japanese, one German
        {"WASTE_ID": [1, 2, 3], "WWTP_NAME": ["Mer", "Tokyo", "Köln"], "CNTRY_ISO": ["AUS", "JPN", "DEU"],
         "LAT_WWTP": [-9.9124, 35.6, 50.9], "LON_WWTP": [144.0526, 139.8, 6.9], "POP_SERVED": [500, 1e6, 1e6]}
    ).to_csv(tmp_path / "hydrowaste.csv", index=False)  # fmt: skip
    return {
        "request_timeout_seconds": 5,
        "request_headers": {"User-Agent": "test"},
        "uncovered": {"countries": ["AU", "JP", "NZ"], "hydrowaste_columns": HW_COLUMNS},
        "page_size": 2,
        "peek_rows": 2,
        "match_distance_m": 1000.0,
        "coordinate_names": {"lon": ["longitude", "long", "lon"], "lat": ["latitude", "lat"]},
        "sources": sources,
        "registers": registers or {},
        "paths": {
            "validation_dir": str(tmp_path / "validation_data"),
            "output_dir": str(tmp_path / "validation_data" / "national_data"),
            "annotated_all_filepath": str(tmp_path / "ours.gpkg"),
            "hydrowaste": str(tmp_path / "hydrowaste.csv"),
        },
    }


def test_fetch_saves_every_kind_keeps_existing_files_and_records_failures(tmp_path, monkeypatch):
    session = Session()
    monkeypatch.setattr(nd, "requests_session_with_retries", lambda: session)
    sources = {
        "a": {"required": True, "peek": True, "files": {"a.csv": {"kind": "http", "url": "https://x.org/a.csv"}}},
        "b": {"required": False, "peek": True, "files": {
            "b.geojson": {"kind": "ogcapi", "url": "https://x.org/collections/c/items"},
            "c.geojson": {"kind": "arcgis", "url": "https://x.org/MapServer/0"},
            "d.zip": {"kind": "http", "url": "https://broken.org/d.zip"},
        }},
    }  # fmt: skip
    cfg = _cfg(tmp_path, sources)
    manifest = nd.fetch_all(cfg)
    assert session.headers == {"User-Agent": "test"}
    root = tmp_path / "validation_data"
    assert (root / "a" / "a.csv").read_bytes() == b"col\n1\n" and manifest["a/a.csv"]["final_url"].endswith("#final")
    assert len(json.loads((root / "b" / "b.geojson").read_text())["features"]) == 3  # both pages
    assert len(json.loads((root / "b" / "c.geojson").read_text())["features"]) == 5  # offsets 0, 2, 4
    assert "403" in manifest["b/d.zip"]["error"] and not (root / "b" / "d.zip").exists()
    assert json.loads((root / "manifest.json").read_text()) == manifest
    calls = len(session.calls)
    nd.fetch_all(cfg)  # only the failed optional file is tried again
    assert len(session.calls) == calls + 1
    sources["b"]["required"] = True
    with pytest.raises(RuntimeError, match=r"b/d.zip"):
        nd.fetch_all(cfg)


def _world(tmp_path):
    """Our plants, a zipped shapefile register and a CSV register."""
    ours = gpd.GeoDataFrame({"WASTE_ID": [1, 2]}, geometry=[Point(10.0, 50.0), Point(11.0, 50.0)], crs=4326)
    ours.to_file(tmp_path / "ours.gpkg", driver="GPKG")
    root = tmp_path / "validation_data"
    (root / "shp").mkdir(parents=True)
    (root / "csv").mkdir()
    reg = gpd.GeoDataFrame(
        {"NAME": ["same", "new far", "area"]},
        geometry=[Point(10.001, 50.0), Point(12.0, 51.0), box(13.0, 52.0, 13.2, 52.2)],
        crs=4326,
    ).to_crs(3035)  # another CRS, and a polygon whose representative point is used
    folder = tmp_path / "reg"
    folder.mkdir()
    reg.iloc[:2].to_file(folder / "plants.shp")
    reg.iloc[2:].to_file(folder / "areas.shp")
    with zipfile.ZipFile(root / "shp" / "plants.zip", "w") as z:
        for f in os.listdir(folder):
            z.write(folder / f, f)
    pd.DataFrame(
        {
            "Nom du système": ["a", "b", "bad"],
            "Latitude (dd)": [50.0, 60.0, 999.0],
            "Longitude (dd)": [11.0005, 20.0, 5.0],
        }
    ).to_csv(root / "csv" / "systems.csv", index=False, encoding="cp1252")  # as Canada's register
    sources = {
        "shp": {"required": False, "peek": True, "files": {"plants.zip": {"kind": "http", "url": "https://x.org/p"}}},
        "csv": {"required": False, "peek": True, "files": {
            "systems.csv": {"kind": "http", "url": "https://x.org/s"},
            "absent.csv": {"kind": "http", "url": "https://x.org/n"},
        }},
    }  # fmt: skip
    registers = {
        "zz_shp": {"source": "shp", "file": "plants.zip", "iso2": "DE", "year": 2020, "encoding": "utf-8"},
        "zz_csv": {"source": "csv", "file": "systems.csv", "iso2": "CA", "year": 2021, "encoding": "cp1252"},
        "zz_absent": {"source": "csv", "file": "absent.csv", "iso2": "CA", "year": 2021, "encoding": "utf-8"},
    }
    return _cfg(tmp_path, sources, registers)


def test_registers_give_the_new_plants_in_one_file(tmp_path):
    cfg = _world(tmp_path)
    summary = nd.build_registers(cfg).set_index("register")
    assert summary.loc["zz_shp", "plants"] == 3 and summary.loc["zz_shp", "new"] == 2
    assert (
        summary.loc["zz_csv", "plants"] == 2 and summary.loc["zz_csv", "new"] == 1
    )  # the row with latitude 999 is dropped
    assert summary.loc["zz_absent", "status"].startswith("FileNotFoundError")
    out = tmp_path / "validation_data" / "national_data"
    new = gpd.read_file(out / "new_plants.gpkg")
    assert sorted(new["register"]) == ["zz_csv", "zz_shp", "zz_shp"] and str(new.crs.to_epsg()) == "4326"
    far = new[new["register"] == "zz_csv"].iloc[0]
    assert (far["lon"], far["lat"]) == (20.0, 60.0) and json.loads(far["attrs"])["Nom du système"] == "b"
    every = gpd.read_file(out / "register_plants.gpkg")
    near = every[~every["new"].astype(bool)]
    assert len(every) == 5 and near["distance_to_ours_m"].max() < 100  # 0.001 and 0.0005 degrees of longitude at 50 N
    assert {"areas.shp:areas", "plants.shp:plants"} == set(every.loc[every["register"] == "zz_shp", "layer"])
    assert pd.read_csv(out / "new_plants.csv")["distance_to_ours_m"].min() > 1000
    with zipfile.ZipFile(out / "results.zip") as z:
        assert set(z.namelist()) == {
            "new_plants.csv",
            "register_plants.csv",
            "uncovered_plants.csv",
            "registers_summary.csv",
            "peek.md",
        }
    listed = pd.read_csv(out / "register_plants.csv")  # every register plant with the nearest plant of ours
    same = listed[~listed["new"]]
    assert len(listed) == 5 and sorted(zip(same["ours_lon"], same["ours_lat"])) == [(10.0, 50.0), (11.0, 50.0)]
    peek = (out / "peek.md").read_text()
    assert "plants.shp" in peek and "NAME" in peek and "Nom du syst" in peek and "absent.csv: not downloaded" in peek


def test_peek_describes_workbooks_and_reports_unreadable_files(tmp_path):
    book = tmp_path / "t.xlsx"
    with pd.ExcelWriter(book) as writer:
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(writer, sheet_name="first", index=False)
        pd.DataFrame({"c": [3]}).to_excel(writer, sheet_name="second", index=False)
    text = nd.peek_file(str(book), 2)
    assert "sheets: ['first', 'second']" in text and "a|b" in text and "1|x" in text
    broken = tmp_path / "broken.geojson"
    broken.write_text("not json")
    assert "could not be described" in nd.peek_file(str(broken), 2)


def test_registers_still_write_every_file_when_no_register_can_be_read(tmp_path):
    cfg = _world(tmp_path)
    cfg["registers"] = {"zz_absent": cfg["registers"]["zz_absent"]}
    summary = nd.build_registers(cfg)
    out = tmp_path / "validation_data" / "national_data"
    assert len(summary) == 1 and len(gpd.read_file(out / "new_plants.gpkg")) == 0
    assert (out / "results.zip").exists() and (out / "new_plants.csv").exists()


def test_uncovered_countries_get_register_plants_and_hydrowaste_records_in_one_file(tmp_path):
    cfg = _cfg(tmp_path, {})
    registers = gpd.GeoDataFrame(
        {"register": ["aus_ga", "aus_ga", "can_wser"], "iso2": ["AU", "AU", "CA"], "lon": [144.05258, 142.61, -113.6],
         "lat": [-9.90973, -9.38, 52.7], "attrs": ['{"feature_name": "Mer"}', '{"feature_name": "Saibai"}', "{}"]},
        geometry=[Point(144.05258, -9.90973), Point(142.61, -9.38), Point(-113.6, 52.7)], crs=4326,
    )  # fmt: skip
    out = nd.build_uncovered(cfg, registers)
    assert sorted(out["iso2"]) == ["AU", "AU", "JP"]  # Canada is covered, Germany too; New Zealand has nothing
    mer = out[out["register_attrs"].str.contains("Mer", na=False)].iloc[0]
    assert mer["sources"] == "register+hydrowaste" and mer["hydrowaste_name"] == "Mer"  # 300 m apart: one plant
    tokyo = out[out["iso2"] == "JP"].iloc[0]
    assert tokyo["sources"] == "hydrowaste" and json.loads(tokyo["hydrowaste_attrs"])["POP_SERVED"] == "1000000.0"
    assert out.loc[out["register_attrs"].str.contains("Saibai", na=False), "sources"].iloc[0] == "register"
    assert "size" not in " ".join(out.columns)  # no size column: the size comes from a model later
