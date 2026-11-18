"""OSM features of every annotation tile from one OSM planet file, without Overpass servers.

Settings: config.yaml, section ``osm_planet``. Commands:
``python -m wwtp.annotation_scripts.osm_planet prepare`` (once) and ``... tiles`` (per plant set).

``prepare`` downloads the planet (the first of ``planet_urls`` that works; resumed after an
interruption, checked against its ``.md5``), keeps its objects with the tags NEW_02 asks Overpass for
(``osmium tags-filter``, with the nodes and member ways they need) in ``filtered.osm.pbf``, and writes
``planet_info.json`` (source, md5, OSM data timestamp, osmium version). The planet is deleted afterwards
unless ``keep_planet``.

``tiles`` cuts the filtered file to the tiles of ``annotation_points_filepath`` (``osmium extract``,
strategy ``smart``: features crossing a tile edge stay whole), and writes per tile what NEW_02's
Overpass query returned: the ways and relations with those tags that cross the tile, the member ways
of those relations, as ``idx_<idx>_lines.geojson`` / ``idx_<idx>_polygons.geojson`` (closed way:
polygon, else line; relation: polygon of its first closed outer way), or ``idx_<idx>_empty.txt``.
Every tile is written again, so all tiles of a set come from the same OSM snapshot.

osmium (osmium-tool) is used from the PATH if it is there; otherwise ``osmium_version`` is installed
once from conda-forge into ``tools_dir`` with micromamba (``micromamba_url``), which needs no conda.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import tarfile
import time
from collections import defaultdict

import geopandas as gpd
import numpy as np
import requests
import shapely
from shapely.geometry import box, mapping

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..utils import atomic_output, configure_logging
from .NEW_02_EXTRACTOSMDATAFULL_GEOJSON import queries, save_tile, tile_files

logger = logging.getLogger(__name__)

PLANET, FILTERED, INFO = "planet-latest.osm.pbf", "filtered.osm.pbf", "planet_info.json"
OPL_ESCAPE = re.compile(r"%([0-9a-fA-F]+)%")
OPL_NODE = re.compile(r"n(-?\d+)x([^y,]*)y([^,]*)")
MAX_TILE_DEGREES = 1.0  # a wider tile crosses the antimeridian: its box would cover the whole world


# --------------------------------------------------------------------------- #
# osmium                                                                      #
# --------------------------------------------------------------------------- #
def osmium_binary(cfg):
    """osmium from the PATH, else osmium-tool ``osmium_version`` installed once into ``tools_dir``."""
    on_path = shutil.which("osmium")
    if on_path:
        return on_path
    env = os.path.join(cfg["paths"]["tools_dir"], f"osmium-tool-{cfg['osmium_version']}")
    exe = os.path.join(env, "bin", "osmium")
    if os.path.exists(exe):
        return exe
    tools = cfg["paths"]["tools_dir"]
    os.makedirs(tools, exist_ok=True)
    package = os.path.join(tools, os.path.basename(cfg["micromamba_url"]))
    logger.info("installing osmium-tool %s into %s (micromamba from conda-forge)", cfg["osmium_version"], env)
    with requests.get(cfg["micromamba_url"], stream=True, timeout=float(cfg["request_timeout_seconds"])) as r:
        r.raise_for_status()
        with open(package, "wb") as fh:
            shutil.copyfileobj(r.raw, fh)
    with tarfile.open(package) as tar:
        tar.extract("bin/micromamba", os.path.join(tools, "micromamba"), filter="data")
    micromamba = os.path.join(tools, "micromamba", "bin", "micromamba")
    os.chmod(micromamba, 0o755)
    environment = {**os.environ, "MAMBA_ROOT_PREFIX": os.path.join(tools, "micromamba", "root")}
    subprocess.run([micromamba, "create", "-y", "-p", env, "-c", "conda-forge",
                    f"osmium-tool={cfg['osmium_version']}"], check=True, env=environment)  # fmt: skip
    return exe


def run(osmium, *args):
    """One osmium command; its output goes to the log."""
    command = [osmium, *map(str, args)]
    logger.info("$ %s", " ".join(command))
    start = time.monotonic()
    subprocess.run(command, check=True)
    logger.info("  done in %.1f min", (time.monotonic() - start) / 60)


def filter_expressions(pairs):
    """``osmium tags-filter`` expressions for ways and relations with the (key, value) pairs."""
    values = defaultdict(list)
    for key, value in pairs:
        values[key].append(value)
    return [f"wr/{key}" if "" in vals else f"wr/{key}={','.join(vals)}" for key, vals in values.items()]


def matches(tags, pairs):
    """An object NEW_02's query would select by its tags."""
    return any(key in tags and (not value or tags[key] == value) for key, value in pairs)


# --------------------------------------------------------------------------- #
# prepare: planet -> filtered                                                 #
# --------------------------------------------------------------------------- #
def download(urls, path, cfg):
    """The planet from the first URL that works, resumed if a part is there, checked against its md5."""
    part = path + ".part"
    timeout = float(cfg["request_timeout_seconds"])
    chunk = int(cfg["download_chunk_mb"]) * 1024 * 1024
    for url in urls:
        try:
            md5 = requests.get(url + ".md5", timeout=timeout).text.split()[0]
            done = os.path.getsize(part) if os.path.exists(part) else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                r.raise_for_status()
                if done and r.status_code != 206:  # the server ignored the range: start again
                    done = 0
                total = done + int(r.headers.get("Content-Length", 0))
                logger.info("downloading %s (%.1f GB, from %.1f GB)", url, total / 1e9, done / 1e9)
                next_log = done + 5e9
                with open(part, "ab" if done else "wb") as fh:
                    for block in r.iter_content(chunk):
                        fh.write(block)
                        done += len(block)
                        if done >= next_log:
                            logger.info("  %.1f of %.1f GB", done / 1e9, total / 1e9)
                            next_log += 5e9
            digest = hashlib.md5()
            with open(part, "rb") as fh:
                while block := fh.read(chunk):
                    digest.update(block)
            if digest.hexdigest() != md5:
                os.remove(part)  # a mix of two weekly planets, or a broken transfer
                raise ValueError(f"md5 {digest.hexdigest()} is not {md5}")
            os.replace(part, path)
            with atomic_output(path + ".json") as tmp, open(tmp, "w") as fh:
                json.dump({"source": url, "md5": md5}, fh)
            return url, md5
        except Exception as exc:
            logger.warning("%s: %s", url, exc)
    raise RuntimeError(f"the planet could not be downloaded from {urls}")


def prepare(cfg):
    folder = cfg["paths"]["planet_dir"]
    os.makedirs(folder, exist_ok=True)
    filtered, info_path, planet = (os.path.join(folder, n) for n in (FILTERED, INFO, PLANET))
    if os.path.exists(filtered) and os.path.exists(info_path):
        logger.info("%s is there already", filtered)
        return
    osmium = osmium_binary(cfg)
    if not os.path.exists(planet):
        download(list(cfg["planet_urls"]), planet, cfg)
    with open(planet + ".json") as fh:
        source = json.load(fh)
    header = json.loads(subprocess.run([osmium, "fileinfo", "-j", planet], check=True, capture_output=True,
                                       text=True).stdout)["header"]["option"]  # fmt: skip
    expressions = filter_expressions(queries)
    run(osmium, "tags-filter", "--overwrite", "-o", filtered, planet, *expressions)
    version = subprocess.run([osmium, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    info = {**source, "planet_bytes": os.path.getsize(planet),
            "osm_timestamp": header.get("osmosis_replication_timestamp", ""), "filter": expressions,
            "osmium": version, "prepared": time.strftime("%Y-%m-%d %H:%M")}  # fmt: skip
    with atomic_output(info_path) as tmp, open(tmp, "w") as fh:
        json.dump(info, fh, indent=2)
    if not cfg["keep_planet"]:
        os.remove(planet)
    logger.info("planet of %s filtered: %.1f GB", info["osm_timestamp"], os.path.getsize(filtered) / 1e9)


# --------------------------------------------------------------------------- #
# tiles: filtered -> features per tile                                        #
# --------------------------------------------------------------------------- #
def unescape(text):
    return OPL_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)


def opl_tags(field):
    """The tags of an OPL ``T`` field (without the T)."""
    tags = {}
    for pair in field.split(",") if field else []:
        key, _, value = pair.partition("=")
        tags[unescape(key)] = unescape(value)
    return tags


def read_opl(path):
    """Ways (id -> (coordinates, tags)) and relations (id -> (members, tags)) of an OPL file whose ways
    carry their node locations."""
    ways, relations = {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split(" ")
            kind, oid = fields[0][0], int(fields[0][1:])
            parts = {f[0]: f[1:] for f in fields[1:] if f}
            tags = opl_tags(parts.get("T", ""))
            if kind == "w":
                coords = [(float(x), float(y)) if x and y else None for _, x, y in OPL_NODE.findall(parts.get("N", ""))]
                ways[oid] = (coords, tags)
            elif kind == "r":
                members = []
                for m in parts.get("M", "").split(",") if parts.get("M") else []:
                    ref, _, role = m.partition("@")
                    members.append({"type": {"n": "node", "w": "way", "r": "relation"}[ref[0]], "ref": int(ref[1:]),
                                    "role": unescape(role)})  # fmt: skip
                relations[oid] = (members, tags)
    return ways, relations


def tile_boxes(grid):
    """Lon/lat bounds of every tile (as NEW_02's Overpass bbox)."""
    bounds = grid.to_crs(4326).bounds
    wide = (bounds["maxx"] - bounds["minx"]) > MAX_TILE_DEGREES
    if wide.any():
        raise ValueError(f"{int(wide.sum())} tiles are wider than {MAX_TILE_DEGREES} degrees (antimeridian)")
    return bounds


def assign(ways, relations, bounds):
    """Per tile position: the ways and relations NEW_02's query returns (hits, and member ways of the hit
    relations)."""
    way_ids = np.array(list(ways), dtype=np.int64)
    lines = [shapely.linestrings([c for c in ways[w][0] if c]) if sum(1 for c in ways[w][0] if c) >= 2 else None
             for w in way_ids]  # fmt: skip
    ok = np.array([g is not None for g in lines])
    tree = shapely.STRtree([box(*b) for b in bounds[["minx", "miny", "maxx", "maxy"]].to_numpy()])
    wi, ti = tree.query(np.array(lines, dtype=object)[ok], predicate="intersects")
    wi = np.flatnonzero(ok)[wi]
    tiles_of_way = defaultdict(set)
    for w, t in zip(way_ids[wi], ti):
        tiles_of_way[int(w)].add(int(t))
    per_tile = defaultdict(lambda: (set(), set()))
    for w, tiles in tiles_of_way.items():
        if matches(ways[w][1], queries):
            for t in tiles:
                per_tile[t][0].add(w)
    for r, (members, tags) in relations.items():
        if not matches(tags, queries):
            continue
        member_ways = [m["ref"] for m in members if m["type"] == "way" and m["ref"] in ways]
        hit = set().union(*(tiles_of_way.get(w, set()) for w in member_ways)) if member_ways else set()
        for t in hit:
            per_tile[t][1].add(r)
            per_tile[t][0].update(member_ways)
    return per_tile


def elements(way_ids, relation_ids, ways, relations):
    """A tile's objects in the shape of an Overpass ``out geom`` answer (ways, then relations, by id)."""
    out = [{"type": "way", "id": w, "tags": ways[w][1],
            "geometry": [{"lon": c[0], "lat": c[1]} if c else None for c in ways[w][0]]}
           for w in sorted(way_ids)]  # fmt: skip
    out += [{"type": "relation", "id": r, "members": relations[r][0], "tags": relations[r][1]}
            for r in sorted(relation_ids)]  # fmt: skip
    for el in out:
        if not el["tags"]:
            del el["tags"]
    return out


_SHARED = {}


def _write_chunk(chunk):
    """Write the tiles of one chunk (a worker process; the data comes from the parent by fork)."""
    ways, relations, per_tile, folder, stamp = (_SHARED[k] for k in ("ways", "relations", "per_tile", "folder",
                                                                        "stamp"))  # fmt: skip
    for position, idx in chunk:
        for path in tile_files(folder, idx):
            if os.path.exists(path):
                os.remove(path)
        way_ids, relation_ids = per_tile.get(position, ((), ()))
        data = {"osm3s": {"timestamp_osm_base": stamp}, "elements": elements(way_ids, relation_ids, ways, relations)}
        save_tile(data, idx, folder)
    return len(chunk)


def tiles(cfg):
    paths = cfg["paths"]
    planet_dir = paths["planet_dir"]
    with open(os.path.join(planet_dir, INFO)) as fh:
        info = json.load(fh)
    points = paths["annotation_points_filepath"]
    grid = gpd.read_file(os.path.join(paths["annotations_grid_dir"], f"grids_{os.path.basename(points)}"))
    grid = grid[grid.geometry.notna() & ~grid.geometry.is_empty].reset_index(drop=True)
    bounds = tile_boxes(grid)
    work = paths["area_dir"]
    os.makedirs(work, exist_ok=True)
    area, extract, located, opl = (os.path.join(work, n) for n in
                                   ("tiles.geojson", "tiles.osm.pbf", "tiles_located.osm.pbf", "tiles.opl"))  # fmt: skip
    union = shapely.union_all([box(*b) for b in bounds[["minx", "miny", "maxx", "maxy"]].to_numpy()])
    with open(area, "w") as fh:
        json.dump({"type": "Feature", "properties": {}, "geometry": mapping(union)}, fh)

    osmium = osmium_binary(cfg)
    run(osmium, "extract", "--overwrite", "-s", "smart", "-p", area, "-o", extract, os.path.join(planet_dir, FILTERED))
    run(osmium, "add-locations-to-ways", "--overwrite", "--ignore-missing-nodes", "-o", located, extract)
    run(osmium, "cat", "--overwrite", "-t", "way", "-t", "relation", "-f",
        "opl,add_metadata=false,locations_on_ways=true", "-o", opl, located)  # fmt: skip
    ways, relations = read_opl(opl)
    per_tile = assign(ways, relations, bounds)
    logger.info("%d ways and %d relations; %d of %d tiles with features (OSM of %s)", len(ways), len(relations),
                len(per_tile), len(grid), info["osm_timestamp"])  # fmt: skip

    folder = paths["annotations_by_osm_dir"]
    os.makedirs(folder, exist_ok=True)
    _SHARED.update(ways=ways, relations=relations, per_tile=per_tile, folder=folder, stamp=info["osm_timestamp"])
    jobs = list(enumerate(grid["idx"].astype(int)))
    size = int(cfg["tiles_per_task"])
    chunks = [jobs[i : i + size] for i in range(0, len(jobs), size)]
    workers = int(cfg["workers"])
    done = 0
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            for n in pool.imap_unordered(_write_chunk, chunks):
                done += n
                if done % (size * 20) < size:
                    logger.info("%d of %d tiles written", done, len(jobs))
    else:
        for chunk in chunks:
            done += _write_chunk(chunk)
    with atomic_output(os.path.join(folder, "osm_source.json")) as tmp, open(tmp, "w") as fh:
        json.dump({**info, "tiles": len(jobs), "tiles_with_features": len(per_tile)}, fh, indent=2)
    logger.info("%d tiles written to %s", done, folder)


def parse_args():
    parser = argparse.ArgumentParser(description="OSM features per annotation tile from a planet file.")
    parser.add_argument("step", choices=["prepare", "tiles"])
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(script_name="osm_planet", **parse_config_overrides(args=args))
    if args.step == "prepare":
        prepare(cfg)
    else:
        tiles(cfg)


if __name__ == "__main__":
    configure_logging()
    main()
