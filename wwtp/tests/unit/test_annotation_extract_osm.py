"""The OSM download of the annotation tiles (NEW_02) and the endpoint benchmark, against fake Overpass
servers on localhost."""

import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

from wwtp.annotation_scripts import NEW_02_EXTRACTOSMDATAFULL_GEOJSON as n2
from wwtp.annotation_scripts import overpass_benchmark as ob
from wwtp.starter import load_config

pytestmark = pytest.mark.unit

# One tile's OSM: a river (line), an industrial area (closed way) and a landuse relation whose outer
# ring is an untagged way. Tiles whose south edge is >= 10 have nothing.
NODES = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0), 4: (0.0, 1.0), 5: (2.0, 2.0), 6: (3.0, 2.0)}
WAYS = {10: ([5, 6], {"waterway": "river"}), 11: ([1, 2, 3, 4, 1], {"landuse": "industrial"}), 12: ([1, 2, 3, 1], {})}
RELATION = {"type": "relation", "id": 20, "members": [{"type": "way", "ref": 12, "role": "outer"}],
            "tags": {"landuse": "farmland", "type": "multipolygon"}}  # fmt: skip


def answer(mode):
    """The fake server's answer in either query form (the same OSM data)."""
    if mode == "body":
        elements = [{"type": "node", "id": i, "lon": x, "lat": y} for i, (x, y) in NODES.items()]
        elements += [{"type": "way", "id": i, "nodes": n, "tags": t} for i, (n, t) in WAYS.items()]
    else:
        elements = [{"type": "way", "id": i, "nodes": n, "tags": t,
                     "geometry": [{"lon": NODES[k][0], "lat": NODES[k][1]} for k in n]}
                    for i, (n, t) in WAYS.items()]  # fmt: skip
    for el in elements:
        if el["type"] == "way" and not el["tags"]:
            del el["tags"]  # Overpass leaves out an empty tags object
    return {"osm3s": {"timestamp_osm_base": "2026-11-17T00:00:00Z"}, "elements": [*elements, RELATION]}


class FakeOverpass:
    """An Overpass instance on localhost: ``delay`` seconds per answer, ``refuse`` first requests 429."""

    def __init__(self, delay=0.0, refuse=0, status=200):
        self.delay, self.refuse, self.status = delay, refuse, status
        self.requests, self.lock = [], threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(fake.status)
                self.end_headers()
                self.wfile.write(b"Connected as: 1\nRate limit: 0\n2 slots available now.\n")

            def do_POST(self):
                query = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())["data"][0]
                with fake.lock:
                    fake.requests.append((self.headers["User-Agent"], query))
                    refused = len(fake.requests) <= fake.refuse
                time.sleep(fake.delay)
                if refused:
                    self.send_response(429)
                    self.end_headers()
                    return
                south = float(re.search(r"\(([-\d.]+),", query).group(1))
                body = answer("geom" if "out geom" in query else "body")
                if south >= 10:
                    body["elements"] = []
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/interpreter"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def servers():
    made = []

    def make(**kw):
        made.append(FakeOverpass(**kw))
        return made[-1]

    yield make
    for s in made:
        s.close()


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(name, raising=False)


def osm_cfg(endpoints, **kw):
    cfg = load_config(script_name="NEW_02_EXTRACTOSMDATAFULL_GEOJSON")["annotations"]
    cfg.update(overpass_endpoints=endpoints, refused_pause_seconds=0.2, request_timeout_seconds=5, progress_every=5)
    cfg.update(kw)
    return cfg


def tiles(n, empty=()):
    """n tiles; those in ``empty`` lie where the fake server has nothing."""
    return [
        n2.Tile(i, f"{20 if i in empty else 0},{i},{(20 if i in empty else 0) + 0.03},{i + 0.03}") for i in range(n)
    ]


def test_both_query_forms_give_the_same_features():
    for mode in ("body", "geom"):
        q = n2.build_query("1,2,3,4", n2.queries, 180, mode)
        assert q.startswith("[out:json][timeout:180];") and 'way["waterway"="stream"](1,2,3,4);' in q
        assert 'relation["landuse"](1,2,3,4);' in q
    assert "(._;>;);" in n2.build_query("1,2,3,4", n2.queries, 180, "body")
    assert n2.build_query("1,2,3,4", n2.queries, 180, "geom").endswith(
        ")->.hits;\n(way.hits; way(r.hits);)->.ways;\n.ways out geom;\nrel.hits;\nout body;"
    )
    with pytest.raises(ValueError):
        n2.build_query("1,2,3,4", n2.queries, 180, "xml")

    (lb, pb), (lg, pg) = n2.elements_to_gdf(answer("body")), n2.elements_to_gdf(answer("geom"))
    for a, b in ((lb, lg), (pb, pg)):
        assert list(a["id"]) == list(b["id"]) and all(x.equals(y) for x, y in zip(a.geometry, b.geometry))
    assert list(lb["id"]) == [10] and list(pb["id"]) == [11, 12, 20]  # the member way too, as before
    assert pb.set_index("id").loc[20, "landuse"] == "farmland"


def test_ways_without_a_geometry_are_left_out():
    data = {"elements": [{"type": "way", "id": 1, "geometry": [{"lon": 0, "lat": 0}, {"lon": 1, "lat": 1}]},
                         {"type": "way", "id": 2, "geometry": [{"lon": 0, "lat": 0}]},  # one point
                         {"type": "way", "id": 3, "geometry": []}]}  # fmt: skip
    lines, polys = n2.elements_to_gdf(data)
    assert list(lines["id"]) == [1] and polys.empty


def test_fetch_tells_answers_refusals_and_server_errors_apart(servers):
    import requests

    s = servers()
    ok = n2.fetch(requests.Session(), s.url, n2.build_query("0,0,1,1", n2.queries, 10, "geom"), 5)
    assert ok.data is not None and not ok.refused and ok.size > 0

    class Remark:
        status_code, headers, content = 200, {}, b"{}"

        def json(self):
            return {"remark": 'runtime error: Query timed out in "query" at line 3', "elements": []}

    class Session:
        def __init__(self, response=None, exc=None):
            self.response, self.exc = response, exc

        def post(self, *a, **k):
            if self.exc:
                raise self.exc
            return self.response

    assert n2.fetch(Session(Remark()), "u", "q", 1).refused  # a server-side timeout is not an empty tile
    assert n2.fetch(Session(exc=TimeoutError("slow")), "u", "q", 1).refused

    class Bad(Remark):
        status_code = 400

    bad = n2.fetch(Session(Bad()), "u", "q", 1)
    assert bad.data is None and not bad.refused and bad.error == "HTTP 400"


def test_downloader_shares_tiles_by_speed_and_hands_refused_ones_on(servers, tmp_path):
    fast, slow, refusing = servers(delay=0.01), servers(delay=0.25), servers(delay=0.01, refuse=1000)
    endpoints = [n2.Endpoint(fast.url, 2), n2.Endpoint(slow.url, 2), n2.Endpoint(refusing.url, 1)]
    cfg = osm_cfg([], user_agent="test-agent")
    given_up = n2.Downloader(tiles(40, empty={3, 7}), endpoints, cfg, str(tmp_path)).run()

    assert given_up == []
    assert n2.done_tiles(str(tmp_path)) == set(range(40))
    assert os.path.exists(tmp_path / "idx_3_empty.txt") and not os.path.exists(tmp_path / "idx_3_lines.geojson")
    assert os.path.exists(tmp_path / "idx_0_lines.geojson") and os.path.exists(tmp_path / "idx_0_polygons.geojson")
    done = {e.url: e.counts["tiles"] for e in endpoints}
    assert done[fast.url] > 2 * done[slow.url]  # nobody waits for the slow one
    assert done[refusing.url] == 0 and endpoints[2].counts["refused"] >= 1
    assert len(refusing.requests) < 10  # it pauses after every refusal instead of hammering
    assert {ua for ua, _ in fast.requests} == {"test-agent"}


def test_downloader_gives_up_after_max_attempts(servers, tmp_path):
    refusing = servers(refuse=1000)
    cfg = osm_cfg([], user_agent="t", max_attempts=2, refused_pause_seconds=0.05)
    given_up = n2.Downloader(tiles(3), [n2.Endpoint(refusing.url, 1)], cfg, str(tmp_path)).run()
    assert sorted(given_up) == [0, 1, 2] and n2.done_tiles(str(tmp_path)) == set()


def test_main_skips_done_tiles_leaves_out_dead_endpoints_and_fails_on_missing_ones(servers, tmp_path, monkeypatch):
    up, dead = servers(), servers(status=503)
    grid_dir, osm_dir = tmp_path / "grid", tmp_path / "osm"
    grid_dir.mkdir()
    osm_dir.mkdir()
    gpd.GeoDataFrame({"idx": [1, 2, 3]}, geometry=[box(0, 0, 0.03, 0.03), box(1, 0, 1.03, 0.03), None],
                     crs=4326).to_file(grid_dir / "grids_points.gpkg")  # fmt: skip
    (osm_dir / "idx_1_empty.txt").write_text("")
    cfg = load_config(script_name="NEW_02_EXTRACTOSMDATAFULL_GEOJSON")
    cfg["annotations"].update(osm_cfg([{"url": up.url, "workers": 1}, {"url": dead.url, "workers": 1}]))
    cfg["paths"].update(annotation_points_filepath=str(tmp_path / "points.gpkg"), annotations_grid_dir=str(grid_dir),
                        annotations_by_osm_dir=str(osm_dir))  # fmt: skip
    monkeypatch.setattr(n2, "parse_args", lambda: None)
    monkeypatch.setattr(n2, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(n2, "load_config", lambda **kw: cfg)

    n2.main()
    assert n2.done_tiles(str(osm_dir)) == {1, 2} and len(up.requests) == 1 and not dead.requests
    n2.main()  # all done: nothing asked
    assert len(up.requests) == 1

    (osm_dir / "idx_2_lines.geojson").unlink()
    (osm_dir / "idx_2_polygons.geojson").unlink()
    up.refuse = 1000
    cfg["annotations"]["max_attempts"] = 1
    with pytest.raises(RuntimeError, match="1 of 1 tiles got no answer"):
        n2.main()


def test_benchmark_compares_endpoints_and_recommends_the_good_ones(servers, tmp_path, monkeypatch):
    good, slow, refusing = servers(delay=0.01), servers(delay=0.1), servers(refuse=1000)
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    gpd.GeoDataFrame({"idx": range(12)}, geometry=[box(i, 0, i + 0.03, 0.03) for i in range(12)],
                     crs=4326).to_file(grid_dir / "grids_points.gpkg")  # fmt: skip
    cfg = load_config(script_name="overpass_benchmark")
    cfg["annotations"].update(
        candidates=[{"url": u.url, "workers": 2} for u in (good, slow, refusing)], sample_tiles=6, compare_tiles=2,
        refused_pause_seconds=0.05, request_timeout_seconds=5,
    )  # fmt: skip
    cfg["paths"].update(annotation_points_filepath=str(tmp_path / "points.gpkg"), annotations_grid_dir=str(grid_dir),
                        annotations_by_osm_dir=str(tmp_path / "osm"), output_dir=str(tmp_path / "bench"))  # fmt: skip
    monkeypatch.setattr(ob, "parse_args", lambda: None)
    monkeypatch.setattr(ob, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(ob, "load_config", lambda **kw: cfg)

    ob.main()
    table = pd.read_csv(tmp_path / "bench" / "endpoints.csv").set_index("url")
    assert table.loc[good.url, "recommended"] and table.loc[slow.url, "recommended"]
    assert not table.loc[refusing.url, "recommended"] and table.loc[refusing.url, "answered_share"] == 0
    assert table.loc[good.url, "tiles_per_min"] > table.loc[slow.url, "tiles_per_min"]
    assert table.loc[good.url, "agreement"] == 1.0
    summary = (tmp_path / "bench" / "summary.md").read_text()
    assert f"{{url: '{good.url}', workers: 2}}" in summary and "12 tiles still without OSM features" in summary
    assert "finds the same features in 100%" in summary
    requests = pd.read_csv(tmp_path / "bench" / "requests.csv")
    assert len(requests) == 3 * (6 + 2)
    assert (tmp_path / "bench" / "results.zip").exists()


def test_find_bbox_handles_missing_and_invalid_geometries():
    assert n2.find_bbox(None) is None
    assert isinstance(n2.find_bbox(Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])), str)
    assert n2.find_bbox(Point(1, 2)) == "2.0,1.0,2.0,1.0"


def test_clean_columns_normalizes_names_and_values():
    gdf = gpd.GeoDataFrame({"bad col!": [1], "another-col": [None], "geometry": [Point(0, 0)]}, crs=4326)
    out = n2.clean_columns(gdf)
    assert list(out.columns) == ["bad_col_", "another_col", "geometry"] and out["bad_col_"].iloc[0] == "1"
    assert out["another_col"].iloc[0] == ""
