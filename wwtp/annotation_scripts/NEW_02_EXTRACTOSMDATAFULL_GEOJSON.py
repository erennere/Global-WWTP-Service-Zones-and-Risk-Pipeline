"""Download the OSM lines and polygons of every annotation tile from Overpass.

Settings: config.yaml, section ``NEW_02_EXTRACTOSMDATAFULL_GEOJSON``.

Per tile one query (the tags of ``queries``); the result goes to ``idx_<idx>_lines.geojson`` and
``idx_<idx>_polygons.geojson`` in ``annotations_by_osm_dir``, or, when the tile has none of those
features, to an empty marker ``idx_<idx>_empty.txt`` so it is not asked again. Tiles with any of these
files are done.

The tiles wait in one queue. Every endpoint of ``overpass_endpoints`` has ``workers`` threads, each
taking the next tile when its last request is answered: a fast endpoint answers more tiles, a slow one
fewer, nobody waits for a batch. Public Overpass instances are shared: requests carry an identifying
``user_agent``; after a refusal (429, 406, 503, 504, a timeout, or a server-side timeout or memory
error) that endpoint pauses ``refused_pause_seconds`` (longer if the server asks) and the tile goes
back to the queue for another endpoint, at most ``max_attempts`` times per run; tiles still without an
answer make the run fail (run it again: only the missing tiles are asked). Endpoints whose status page
does not answer are left out. ``output_mode`` ``geom`` asks for the way geometry inline instead of
the nodes (``body``, the first global run): the same features, less to send.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon

from ..geo_utils import repair_geometry
from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..utils import atomic_output, configure_logging, requests_session_with_retries

logger = logging.getLogger(__name__)

# (key, value) pairs; a list, not a dict, because one key ("waterway") is
# queried with two values - a dict literal silently kept only the last one,
# so streams were never requested. An empty value means "any value".
queries = [
    ("man_made", "wastewater_plant"),
    ("waterway", "stream"),
    ("waterway", "river"),
    ("landuse", ""),
    ("power", "plant"),
    ("industrial", ""),
]

# Answers after which an endpoint pauses and the tile is asked elsewhere (rate limit, refused,
# overloaded, gateway timeout); a 400 is a bad query and is not retried.
REFUSED_STATUS = {406, 429, 503, 504}
# Overpass answers 200 with a "remark" when the query failed on the server
SERVER_ERROR = re.compile(r"runtime error|timed out|out of memory", re.IGNORECASE)


def clean_columns(gdf):
    """Clean column names and ensure valid types for GeoJSON export.

    Tag keys that clean to the same name, regardless of case (``name:ja`` and ``name_ja``, keys alike in
    their first 60 characters, ``Acres`` and ``ACRES``, which DuckDB takes for one column), get ``_2``,
    ``_3``, ... in the order they come."""
    if gdf.empty:
        return gdf
    names, taken = [], {"geometry"}
    for column in gdf.columns:
        if column == "geometry":
            names.append(column)
            continue
        name = base = re.sub(r"[^0-9a-zA-Z_]", "_", column)[:60]
        n = 1
        while name.lower() in taken:
            n += 1
            name = f"{base}_{n}"
        taken.add(name.lower())
        names.append(name)
    gdf.columns = names
    for col in gdf.columns:
        if col != "geometry":
            gdf[col] = gdf[col].apply(lambda x: str(x) if x is not None else "")
    return gdf


def build_query(bbox, pairs, timeout_seconds, output_mode):
    """Overpass QL for the ways and relations of ``pairs`` in ``bbox``.

    ``body``: the hits with all their nodes and member ways (``(._;>;)``), nodes as separate elements.
    ``geom``: the same ways (hits and member ways of the hit relations) with their coordinates inline,
    then the hit relations with their members; no node elements.
    """
    lines = [f"[out:json][timeout:{int(timeout_seconds)}];", "("]
    for key, value in pairs:
        filter_expr = f'["{key}"="{value}"]' if value else f'["{key}"]'
        lines += [f"  way{filter_expr}({bbox});", f"  relation{filter_expr}({bbox});"]
    if output_mode == "body":
        return "\n".join(lines + [");", "(._;>;);", "out body;"])
    if output_mode == "geom":
        return "\n".join(lines + [")->.hits;", "(way.hits; way(r.hits);)->.ways;", ".ways out geom;",
                                  "rel.hits;", "out body;"])  # fmt: skip
    raise ValueError(f"output_mode must be body or geom, not {output_mode!r}")


@dataclass
class Answer:
    """One Overpass request: ``data`` when it worked, else whether the endpoint refused it."""

    data: dict | None
    status: int | None
    seconds: float
    size: int = 0
    refused: bool = False
    retry_after: float = 0.0
    error: str = ""


def fetch(session, url, query, timeout_seconds):
    """POST one query; never raises for network or server errors."""
    start = time.monotonic()
    try:
        r = session.post(url, data={"data": query}, timeout=timeout_seconds)
    except Exception as exc:  # timeouts and connection errors count as refusals
        return Answer(None, None, time.monotonic() - start, refused=True, error=f"{type(exc).__name__}: {exc}")
    seconds = time.monotonic() - start
    retry_after = r.headers.get("Retry-After", "")
    retry_after = float(retry_after) if retry_after.strip().isdigit() else 0.0
    if r.status_code != 200:
        return Answer(None, r.status_code, seconds, len(r.content), r.status_code in REFUSED_STATUS, retry_after,
                      f"HTTP {r.status_code}")  # fmt: skip
    try:
        data = r.json()
    except (json.JSONDecodeError, ValueError):
        return Answer(None, 200, seconds, len(r.content), True, error="not JSON")
    remark = data.get("remark", "")
    if SERVER_ERROR.search(remark):
        return Answer(None, 200, seconds, len(r.content), True, error=remark[:200])
    return Answer(data, 200, seconds, len(r.content))


def elements_to_gdf(data):
    """Lines and polygons of an Overpass answer (``out body`` with nodes, or ``out geom``).

    Every way is a feature (closed: polygon, else line); a relation is a polygon from its first closed
    outer way.
    """
    if not data or "elements" not in data:
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return empty.copy(), empty.copy()
    elements = data["elements"]
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in elements if el["type"] == "node"}

    def coords_of(way):
        if "geometry" in way:
            return [(p["lon"], p["lat"]) for p in way["geometry"] if p]
        return [nodes[n] for n in way.get("nodes", []) if n in nodes]

    ways = {el["id"]: coords_of(el) for el in elements if el["type"] == "way"}
    lns, polys = [], []
    for el in elements:
        # the OSM id as the first column, so it keeps the name ``id`` next to a tag ``ID`` (clean_columns);
        # its value wins over a tag ``id``
        tags = {"id": el["id"], **el.get("tags", {})}
        if el["type"] == "way":
            coords = ways[el["id"]]
            try:  # a degenerate way (one point, a closed way of three) is left out
                if coords[0] == coords[-1]:
                    polys.append({**tags, "id": el["id"], "geometry": Polygon(coords)})
                else:
                    lns.append({**tags, "id": el["id"], "geometry": LineString(coords)})
            except (IndexError, ValueError):
                continue
        elif el["type"] == "relation":
            outers = [
                ways[m["ref"]]
                for m in el.get("members", [])
                if m.get("type") == "way" and m.get("role") == "outer" and ways.get(m["ref"])
            ]
            outers = [c for c in outers if c[0] == c[-1]]
            if outers:
                try:
                    polys.append({**tags, "id": el["id"], "geometry": Polygon(outers[0])})
                except Exception:
                    pass

    def frame(rows):
        if rows:
            return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    return frame(lns), frame(polys)


def find_bbox(geometry):
    """Return an Overpass bbox string for a geometry, or None when invalid."""
    if geometry is None or pd.isna(geometry):
        return None
    geometry = repair_geometry(geometry)
    if geometry is None:
        return None
    minx, miny, maxx, maxy = geometry.bounds
    return f"{miny},{minx},{maxy},{maxx}"


def tile_files(output_folder, idx):
    """Lines, polygons and empty-marker paths of one tile."""
    return tuple(os.path.join(output_folder, f"idx_{idx}_{kind}") for kind in
                 ("lines.geojson", "polygons.geojson", "empty.txt"))  # fmt: skip


def done_tiles(output_folder):
    """Tile ids with any of their output files."""
    pattern = re.compile(r"^idx_(-?\d+)_(?:lines\.geojson|polygons\.geojson|empty\.txt)$")
    return {int(m.group(1)) for name in os.listdir(output_folder) if (m := pattern.match(name))}


def save_tile(data, idx, output_folder):
    """Write a tile's lines and polygons, or its empty marker. Returns (lines, polygons)."""
    lines, polys = elements_to_gdf(data)
    line_path, poly_path, empty_path = tile_files(output_folder, idx)
    for gdf, path in ((lines, line_path), (polys, poly_path)):
        if not gdf.empty:
            with atomic_output(path) as tmp:
                clean_columns(gdf).to_file(tmp, driver="GeoJSON")
    if lines.empty and polys.empty:
        with atomic_output(empty_path) as tmp, open(tmp, "w") as fh:
            fh.write(data.get("osm3s", {}).get("timestamp_osm_base", ""))
    return len(lines), len(polys)


def status_url(url):
    """The status page next to an interpreter URL."""
    return re.sub(r"/interpreter/?$", "/status", url)


@dataclass
class Endpoint:
    url: str
    workers: int
    resume_at: float = 0.0
    counts: collections.Counter = field(default_factory=collections.Counter)
    seconds: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def count(self, key, seconds=0.0):
        with self.lock:
            self.counts[key] += 1
            self.seconds += seconds

    def pause(self, seconds):
        with self.lock:
            self.resume_at = max(self.resume_at, time.monotonic() + seconds)

    def wait(self):
        while (delay := self.resume_at - time.monotonic()) > 0:
            time.sleep(min(delay, 5.0))


@dataclass
class Tile:
    idx: int
    bbox: str
    attempts: int = 0
    refused_by: set = field(default_factory=set)


class Downloader:
    """The tile queue and one worker pool per endpoint."""

    def __init__(self, tiles, endpoints, cfg, output_folder):
        self.queue = collections.deque(tiles)
        self.endpoints = endpoints
        self.cfg = cfg
        self.output_folder = output_folder
        self.lock = threading.Lock()
        self.in_flight = 0
        self.finished = 0
        self.given_up = []
        self.total = len(tiles)
        self.start = time.monotonic()

    def take(self, endpoint):
        """The next tile, preferring one this endpoint has not refused; None when all are done."""
        while True:
            with self.lock:
                for _ in range(len(self.queue)):
                    tile = self.queue.popleft()
                    others = len(tile.refused_by | {endpoint.url}) < len(self.endpoints)
                    if endpoint.url in tile.refused_by and others:
                        self.queue.append(tile)
                        continue
                    self.in_flight += 1
                    return tile
                if not self.queue and not self.in_flight:
                    return None
            time.sleep(1.0)  # tiles in flight may come back

    def give_back(self, tile, endpoint, answer):
        tile.attempts += 1
        tile.refused_by.add(endpoint.url)
        with self.lock:
            self.in_flight -= 1
            if tile.attempts < int(self.cfg["max_attempts"]):
                self.queue.append(tile)
            else:
                self.given_up.append(tile.idx)
                logger.warning("tile %s: given up after %d attempts (%s)", tile.idx, tile.attempts, answer.error)

    def done(self, tile, failed=False):
        with self.lock:
            self.in_flight -= 1
            if failed:
                self.given_up.append(tile.idx)
                return
            self.finished += 1
            finished = self.finished
        if finished % int(self.cfg["progress_every"]) == 0:
            self.report()

    def work(self, endpoint):
        """One worker of an endpoint: ask for tiles until none is left."""
        session = requests_session_with_retries(total_retries=0)  # one session per thread
        session.headers["User-Agent"] = self.cfg["user_agent"]
        timeout_seconds = float(self.cfg["request_timeout_seconds"])
        while True:
            endpoint.wait()
            tile = self.take(endpoint)
            if tile is None:
                return
            try:
                query = build_query(tile.bbox, queries, self.cfg["query_timeout_seconds"], self.cfg["output_mode"])
                answer = fetch(session, endpoint.url, query, timeout_seconds)
                if answer.data is not None:
                    save_tile(answer.data, tile.idx, self.output_folder)
                    endpoint.count("tiles", answer.seconds)
                    self.done(tile)
                elif answer.refused:
                    endpoint.count("refused", answer.seconds)
                    endpoint.pause(max(float(self.cfg["refused_pause_seconds"]), answer.retry_after))
                    self.give_back(tile, endpoint, answer)
                else:  # a bad query: no other endpoint would answer it either
                    endpoint.count("failed", answer.seconds)
                    logger.error("tile %s: %s at %s", tile.idx, answer.error, endpoint.url)
                    self.done(tile, failed=True)
            except Exception as exc:  # one tile that cannot be written must not stop the others
                endpoint.count("failed")
                logger.error("tile %s: %s: %s", tile.idx, type(exc).__name__, exc)
                self.done(tile, failed=True)

    def report(self):
        minutes = (time.monotonic() - self.start) / 60
        per_endpoint = ", ".join(
            f"{e.url.split('/')[2]}: {e.counts['tiles']} ({e.counts['tiles'] / max(minutes, 1e-9):.1f}/min, "
            f"{e.counts['refused']} refused)"
            for e in self.endpoints
        )
        logger.info("%d of %d tiles in %.1f min (%.1f/min); %s", self.finished, self.total, minutes,
                    self.finished / max(minutes, 1e-9), per_endpoint)  # fmt: skip

    def run(self):
        threads = [
            threading.Thread(target=self.work, args=(e,), name=f"{e.url.split('/')[2]}-{i}", daemon=True)
            for e in self.endpoints
            for i in range(int(e.workers))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.report()
        return self.given_up


def reachable(endpoints, user_agent, timeout_seconds):
    """The endpoints whose status page answers; logs each status (rate limit, free slots)."""
    session = requests_session_with_retries(total_retries=0)
    session.headers["User-Agent"] = user_agent
    alive = []
    for e in endpoints:
        try:
            r = session.get(status_url(e.url), timeout=timeout_seconds)
            ok = r.status_code == 200
            logger.info("%s status %s: %s", e.url, r.status_code, " | ".join(r.text.splitlines()[:4]))
        except Exception as exc:
            ok = False
            logger.warning("%s: status page not reached (%s)", e.url, exc)
        if ok:
            alive.append(e)
        else:
            logger.warning("%s left out: its status page does not answer", e.url)
    if not alive:
        raise RuntimeError("no Overpass endpoint answers; see the log")
    return alive


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run NEW_02_EXTRACTOSMDATAFULL_GEOJSON.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Download the OSM features of every tile that has none yet."""
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="NEW_02_EXTRACTOSMDATAFULL_GEOJSON", **overrides)
    osm = cfg["annotations"]
    points_path = cfg["paths"]["annotation_points_filepath"]
    grid_filepath = os.path.join(cfg["paths"]["annotations_grid_dir"], f"grids_{os.path.basename(points_path)}")
    output_folder = cfg["paths"]["annotations_by_osm_dir"]
    os.makedirs(output_folder, exist_ok=True)

    grid = gpd.read_file(grid_filepath).to_crs(4326)
    grid["bbox"] = grid["geometry"].map(find_bbox)
    grid = grid[grid["bbox"].notna()]
    if not cfg["overwrite_existing"]:
        grid = grid[~grid["idx"].astype(int).isin(done_tiles(output_folder))]
    if grid.empty:
        logger.info("every tile has its OSM features already")
        return
    tiles = [Tile(int(i), b) for i, b in zip(grid["idx"], grid["bbox"])]
    endpoints = [Endpoint(e["url"], int(e["workers"])) for e in osm["overpass_endpoints"]]
    endpoints = reachable(endpoints, osm["user_agent"], float(osm["request_timeout_seconds"]))
    logger.info("%d tiles to download from %s", len(tiles), ", ".join(f"{e.url} ({e.workers})" for e in endpoints))
    given_up = Downloader(tiles, endpoints, osm, output_folder).run()
    if given_up:
        raise RuntimeError(f"{len(given_up)} of {len(tiles)} tiles got no answer (e.g. {sorted(given_up)[:10]}); "
                           "run again: only the missing tiles are asked")  # fmt: skip


if __name__ == "__main__":
    configure_logging()
    main()
