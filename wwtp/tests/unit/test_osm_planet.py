"""OSM features per tile from a planet file (osm_planet): the same features as NEW_02's Overpass query."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from wwtp.annotation_scripts import osm_planet as op
from wwtp.annotation_scripts.NEW_01_GENERATEGRIDS import square_tiles
from wwtp.starter import load_config

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(name, raising=False)


def test_the_filter_asks_for_the_tags_of_the_overpass_query():
    assert op.filter_expressions(op.queries) == [
        "wr/man_made=wastewater_plant", "wr/waterway=stream,river", "wr/landuse", "wr/power=plant", "wr/industrial"]  # fmt: skip
    assert op.matches({"waterway": "river"}, op.queries) and op.matches({"landuse": "x"}, op.queries)
    assert not op.matches({"waterway": "canal"}, op.queries) and not op.matches({}, op.queries)


def test_opl_lines_with_escapes_and_missing_nodes(tmp_path):
    (tmp_path / "f.opl").write_text(
        "w3 Tname=a%20%b%2c%c%3d%d,landuse=x Nn1x2y1,n2x2.1y1.5,n99xy\n"
        "w4 T Nn1x2y1\n"
        "r5 Ttype=multipolygon,landuse=y Mw3@outer%20%x,n1@,w4@inner\n"
        "r6 T M\n",
        encoding="utf-8",
    )  # fmt: skip
    ways, relations = op.read_opl(tmp_path / "f.opl")
    assert ways[3] == ([(2.0, 1.0), (2.1, 1.5), None], {"name": "a b,c=d", "landuse": "x"})
    assert ways[4] == ([(2.0, 1.0)], {})
    assert relations[5][0] == [{"type": "way", "ref": 3, "role": "outer x"}, {"type": "node", "ref": 1, "role": ""},
                               {"type": "way", "ref": 4, "role": "inner"}]  # fmt: skip
    assert relations[6] == ([], {})


def test_tiles_get_the_hits_and_the_member_ways_of_hit_relations():
    ways = {
        1: ([(0.0, 0.0), (0.2, 0.0)], {"waterway": "river"}),  # crosses tile 0
        2: ([(0.01, 0.01), (0.02, 0.01), (0.02, 0.02), (0.01, 0.01)], {"highway": "x"}),  # not asked for
        3: ([(1.01, 0.01), (1.02, 0.01), (1.02, 0.02), (1.01, 0.01)], {}),  # outer of relation 7, tile 1
        4: ([(5.0, 5.0), (5.1, 5.0)], {}),  # inner far away: comes with the relation
        5: ([(0.0, 0.0)], {"landuse": "x"}),  # one node: no line
    }
    relations = {
        7: (
            [
                {"type": "way", "ref": 3, "role": "outer"},
                {"type": "way", "ref": 4, "role": "inner"},
                {"type": "way", "ref": 99, "role": "outer"},
            ],
            {"landuse": "farmland"},
        ),  # fmt: skip
        8: ([{"type": "way", "ref": 1, "role": ""}], {"route": "canoe"}),
    }  # not asked for
    bounds = pd.DataFrame({"minx": [-0.05, 0.95, 3.0], "miny": [-0.05, -0.05, 3.0], "maxx": [0.05, 1.05, 3.1],
                           "maxy": [0.05, 0.05, 3.1]})  # fmt: skip
    per_tile = op.assign(ways, relations, bounds)
    assert per_tile[0] == ({1}, set()) and per_tile[1] == ({3, 4}, {7}) and 2 not in per_tile
    els = op.elements(*per_tile[1], ways, relations)
    assert [(e["type"], e["id"]) for e in els] == [("way", 3), ("way", 4), ("relation", 7)]
    assert "tags" not in els[0] and els[2]["tags"] == {"landuse": "farmland"}


class FileServer:
    """Serves ``files`` (name -> bytes) on localhost, with byte ranges if ``ranges``."""

    def __init__(self, files, ranges=True):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                name = self.path.lstrip("/")
                if name not in files:
                    self.send_response(404)
                    self.end_headers()
                    return
                body, start = files[name], 0
                if ranges and self.headers["Range"]:
                    start = int(self.headers["Range"].split("=")[1].rstrip("-"))
                    self.send_response(206)
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body) - start))
                self.end_headers()
                self.wfile.write(body[start:])
                server.requests.append((name, start))

        self.requests = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self):
        self.httpd.shutdown()


def planet_cfg(tmp_path, **paths):
    cfg = load_config(script_name="osm_planet")
    cfg["download_chunk_mb"], cfg["request_timeout_seconds"] = 1, 5
    cfg["paths"].update(planet_dir=str(tmp_path / "planet"), tools_dir=str(tmp_path / "tools"), **paths)
    return cfg


def test_download_resumes_checks_the_md5_and_falls_back_to_the_next_url(tmp_path):
    body = os.urandom(3 * 1024 * 1024 + 17)
    md5 = hashlib.md5(body).hexdigest()
    good = FileServer({"p.pbf": body, "p.pbf.md5": f"{md5}  p.pbf\n".encode()})
    wrong = FileServer({"p.pbf": body[:-1] + b"x", "p.pbf.md5": f"{md5}  p.pbf\n".encode()})
    try:
        cfg = planet_cfg(tmp_path)
        path = str(tmp_path / "planet.pbf")
        (tmp_path / "planet.pbf.part").write_bytes(body[:1000])  # an interrupted download
        url, got = op.download([f"{wrong.url}/p.pbf", f"{good.url}/p.pbf"], path, cfg)
        assert url == f"{good.url}/p.pbf" and got == md5
        assert open(path, "rb").read() == body and not os.path.exists(path + ".part")
        assert ("p.pbf", 1000) in wrong.requests  # resumed there, then thrown away (wrong md5) ...
        assert ("p.pbf", 0) in good.requests  # ... and downloaded whole from the next mirror
        assert json.load(open(path + ".json")) == {"source": url, "md5": md5}
        with pytest.raises(RuntimeError, match="could not be downloaded"):
            op.download([f"{good.url}/missing.pbf"], str(tmp_path / "x.pbf"), cfg)
    finally:
        good.close()
        wrong.close()


def test_a_server_without_byte_ranges_sends_the_whole_file_again(tmp_path):
    body = os.urandom(200_000)
    server = FileServer({"p.pbf": body, "p.pbf.md5": hashlib.md5(body).hexdigest().encode()}, ranges=False)
    try:
        path = str(tmp_path / "planet.pbf")
        (tmp_path / "planet.pbf.part").write_bytes(body[:500])
        op.download([f"{server.url}/p.pbf"], path, planet_cfg(tmp_path))
        assert open(path, "rb").read() == body
    finally:
        server.close()


def offset(lon, lat, east_m=0.0, north_m=0.0):
    return lon + east_m / (111_320 * math.cos(math.radians(lat))), lat + north_m / 110_574


def synthetic_planet(path):
    """A tiny planet: a river and an industrial area at (8, 47), a farmland relation with an untagged
    outer way and a wastewater plant at (9, 47), a road, and a forest far from every tile."""
    nodes, ways = [], []

    def node(lon, lat):
        nodes.append((len(nodes) + 1, lon, lat))
        return len(nodes)

    def ring(lon, lat, corners):
        ids = [node(*offset(lon, lat, x, y)) for x, y in corners]
        return ids + ids[:1]

    ways.append((10, [node(*offset(8, 47, -3000)), node(*offset(8, 47, 0, 100)), node(*offset(8, 47, 3000))],
                 {"waterway": "river", "name": "Rhein, the river"}))  # fmt: skip
    ways.append((11, ring(8, 47, [(100, 100), (400, 100), (400, 400), (100, 400)]), {"landuse": "industrial"}))
    ways.append((12, ring(8, 47, [(-500, -500), (-200, -500), (-200, -200)]), {"highway": "service"}))
    ways.append((13, ring(9, 47, [(0, 0), (300, 0), (300, 300), (0, 300)]), {}))
    ways.append((14, ring(9, 47, [(500, 500), (700, 500), (700, 700)]), {"man_made": "wastewater_plant"}))
    ways.append((15, ring(30, 0, [(0, 0), (100, 0), (100, 100)]), {"landuse": "forest"}))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="test">']
    lines += [f'<node id="{i}" lat="{la:.7f}" lon="{lo:.7f}" version="1"/>' for i, lo, la in nodes]
    for i, nds, tags in ways:
        lines.append(f'<way id="{i}" version="1">' + "".join(f'<nd ref="{n}"/>' for n in nds)
                     + "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items()) + "</way>")  # fmt: skip
    lines.append('<relation id="20" version="1"><member type="way" ref="13" role="outer"/>'
                 '<tag k="type" v="multipolygon"/><tag k="landuse" v="farmland"/></relation>')  # fmt: skip
    lines.append("</osm>")
    xml = path.with_suffix(".osm")
    xml.write_text("\n".join(lines))
    subprocess.run([shutil.which("osmium"), "cat", str(xml), "-o", str(path), "--overwrite",
                    "--output-header=osmosis_replication_timestamp=2026-11-16T00:00:00Z"], check=True)  # fmt: skip


@pytest.mark.skipif(shutil.which("osmium") is None, reason="osmium-tool is not installed")
def test_planet_to_tiles_gives_overpass_features(tmp_path):
    planet = tmp_path / "serve" / "planet-latest.osm.pbf"
    planet.parent.mkdir()
    synthetic_planet(planet)
    body = planet.read_bytes()
    server = FileServer(
        {"planet-latest.osm.pbf": body, "planet-latest.osm.pbf.md5": hashlib.md5(body).hexdigest().encode()}
    )
    grid_dir, osm_dir = tmp_path / "grid", tmp_path / "osm"
    grid_dir.mkdir()
    osm_dir.mkdir()
    (osm_dir / "idx_6_lines.geojson").write_text("{}")  # an older download of tile 6: written again
    points = gpd.GeoDataFrame({"idx": [5, 6, 7]}, geometry=[Point(8, 47), Point(9, 47), Point(20, 10)], crs=4326)
    square_tiles(points, 3072 * 1.194 / 2).to_file(grid_dir / "grids_points.gpkg")
    cfg = planet_cfg(tmp_path, annotation_points_filepath=str(tmp_path / "points.gpkg"),
                     annotations_grid_dir=str(grid_dir), annotations_by_osm_dir=str(osm_dir),
                     area_dir=str(tmp_path / "area"))  # fmt: skip
    cfg["planet_urls"] = [f"{server.url}/planet-latest.osm.pbf"]
    cfg["workers"], cfg["tiles_per_task"] = 2, 1
    try:
        op.prepare(cfg)
        op.tiles(cfg)
    finally:
        server.close()

    info = json.load(open(tmp_path / "planet" / "planet_info.json"))
    assert info["osm_timestamp"] == "2026-11-16T00:00:00Z" and info["md5"] == hashlib.md5(body).hexdigest()
    assert not (tmp_path / "planet" / "planet-latest.osm.pbf").exists()  # keep_planet false
    assert sorted(f for f in os.listdir(osm_dir) if f.startswith("idx_")) == [
        "idx_5_lines.geojson", "idx_5_polygons.geojson", "idx_6_polygons.geojson", "idx_7_empty.txt"]  # fmt: skip
    lines5 = gpd.read_file(osm_dir / "idx_5_lines.geojson")
    assert list(lines5["id"]) == ["10"] and lines5["name"][0] == "Rhein, the river"
    assert list(gpd.read_file(osm_dir / "idx_5_polygons.geojson")["id"]) == ["11"]  # not the road
    polys6 = gpd.read_file(osm_dir / "idx_6_polygons.geojson")
    assert list(polys6["id"]) == ["13", "14", "20"]  # the untagged outer way too, as from Overpass
    assert json.load(open(osm_dir / "osm_source.json"))["tiles_with_features"] == 2
