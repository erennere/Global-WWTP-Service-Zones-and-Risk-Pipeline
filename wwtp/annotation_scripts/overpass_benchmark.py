"""Overpass endpoints compared on a sample of the annotation tiles, to choose the ones NEW_02 uses.

Settings: config.yaml, section ``overpass_benchmark``. Command:
``python -m wwtp.annotation_scripts.overpass_benchmark`` (Snakemake target ``overpass_benchmark``; it
needs internet, which the compute nodes have).

Every endpoint of ``candidates`` gets the same ``sample_tiles`` random tiles (NEW_01's square around a
random plant of ``annotation_points_filepath``) with NEW_02's query, with
its ``workers`` requests at a time; the endpoints run side by side (each is a different server). The
first ``compare_tiles`` are then asked once more with the other ``output_mode``, one at a time, to
compare the two query forms. Per request: answer, seconds, bytes, features and the age of the data.
An endpoint's tiles agree when it finds the same features as most endpoints.

Writes to ``paths.output_dir``: ``requests.csv`` (one row per request), ``endpoints.csv`` (per endpoint:
answered share, refusals, median and 90th-percentile seconds, tiles per minute, data timestamp,
agreement), ``summary.md`` (the same, the recommended endpoints for NEW_02 and the time the remaining
tiles would take) and ``results.zip`` with all three to send back.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import queue
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..utils import atomic_output, configure_logging, requests_session_with_retries
from .NEW_01_GENERATEGRIDS import check_idx, square_tiles
from .NEW_02_EXTRACTOSMDATAFULL_GEOJSON import Tile, build_query, done_tiles, elements_to_gdf, fetch, find_bbox, queries

logger = logging.getLogger(__name__)
OTHER_MODE = {"geom": "body", "body": "geom"}


def features_key(data):
    """Counts and a fingerprint of the features an answer gives (ids per kind)."""
    lines, polys = elements_to_gdf(data)
    ids = ",".join(f"l{i}" for i in sorted(lines["id"])) + ";" + ",".join(f"p{i}" for i in sorted(polys["id"])) \
        if len(lines) or len(polys) else ""  # fmt: skip
    return len(lines), len(polys), hashlib.sha1(ids.encode()).hexdigest()[:12]


def ask(session, url, tile, mode, cfg):
    """One request, as a result row."""
    query = build_query(tile.bbox, queries, cfg["query_timeout_seconds"], mode)
    answer = fetch(session, url, query, float(cfg["request_timeout_seconds"]))
    row = {"url": url, "idx": tile.idx, "mode": mode, "status": answer.status, "seconds": round(answer.seconds, 2),
           "bytes": answer.size, "ok": answer.data is not None, "refused": answer.refused, "error": answer.error,
           "lines": np.nan, "polygons": np.nan, "features": "", "osm_base": ""}  # fmt: skip
    if answer.data is not None:
        row["lines"], row["polygons"], row["features"] = features_key(answer.data)
        row["osm_base"] = answer.data.get("osm3s", {}).get("timestamp_osm_base", "")
    return row


def run_endpoint(endpoint, tiles, cfg):
    """The sample at the endpoint's concurrency, then the comparison tiles in the other query form."""
    sessions = queue.Queue()  # one session per concurrent request
    for _ in range(int(endpoint["workers"])):
        s = requests_session_with_retries(total_retries=0)
        s.headers["User-Agent"] = cfg["user_agent"]
        sessions.put(s)
    pause = float(cfg["refused_pause_seconds"])
    resume = {"at": 0.0}
    lock = threading.Lock()

    def one(tile, mode):
        while (delay := resume["at"] - time.monotonic()) > 0:
            time.sleep(delay)
        s = sessions.get()
        try:
            row = ask(s, endpoint["url"], tile, mode, cfg)
        finally:
            sessions.put(s)
        if row["refused"]:
            with lock:
                resume["at"] = max(resume["at"], time.monotonic() + pause)
        return row

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=int(endpoint["workers"])) as pool:
        rows = list(pool.map(lambda t: one(t, cfg["output_mode"]), tiles))
    wall = time.monotonic() - start
    for row in rows:
        row["workers"], row["wall_seconds"] = int(endpoint["workers"]), wall
    other = OTHER_MODE[cfg["output_mode"]]
    for tile in tiles[: int(cfg["compare_tiles"])]:
        row = one(tile, other)
        row["workers"], row["wall_seconds"] = 1, np.nan
        rows.append(row)
    logger.info("%s: %d of %d answered in %.0f s", endpoint["url"], sum(r["ok"] for r in rows[: len(tiles)]),
                len(tiles), wall)  # fmt: skip
    return rows


def markdown_table(frame):
    """A DataFrame as a Markdown table (numbers with two decimals)."""

    def cell(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    return "\n".join(
        [head, rule, *("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False))]
    )


def summarize(requests, cfg):
    """Per-endpoint table, the recommended endpoints and the expected time for the remaining tiles."""
    main = requests[requests["mode"] == cfg["output_mode"]].copy()
    ok = main[main["ok"]]
    consensus = ok.groupby("idx")["features"].agg(lambda s: s.value_counts().index[0])
    main["agrees"] = main["ok"] & (main["features"] == main["idx"].map(consensus))
    table = []
    for url, g in main.groupby("url", sort=False):
        answered = g[g["ok"]]
        minutes = g["wall_seconds"].iloc[0] / 60
        table.append({
            "url": url, "workers": int(g["workers"].iloc[0]), "tiles": len(g), "answered_share": g["ok"].mean(),
            "refused": int(g["refused"].sum()), "median_s": answered["seconds"].median(),
            "p90_s": answered["seconds"].quantile(0.9), "tiles_per_min": len(answered) / max(minutes, 1e-6),
            "median_kb": answered["bytes"].median() / 1024, "osm_base": answered["osm_base"].max(),
            "agreement": g.loc[g["ok"], "agrees"].mean() if len(answered) else np.nan,
            "errors": "; ".join(sorted(set(g.loc[~g["ok"], "error"]))[:3]),
        })  # fmt: skip
    table = pd.DataFrame(table)
    good = (table["answered_share"] >= float(cfg["min_answered_share"])) & (
        table["agreement"] >= float(cfg["min_agreement"])
    )
    table["recommended"] = good
    best = table[good].sort_values("tiles_per_min", ascending=False)
    rate = best["tiles_per_min"].sum()

    other = requests[requests["mode"] != cfg["output_mode"]]
    pair = other.merge(main, on=["url", "idx"], suffixes=("_other", ""))
    pair = pair[pair["ok"] & pair["ok_other"]]
    compare = {
        "tiles": len(pair),
        "same_features": float((pair["features"] == pair["features_other"]).mean()) if len(pair) else np.nan,
        "seconds_ratio": float(pair["seconds"].sum() / pair["seconds_other"].sum()) if len(pair) else np.nan,
        "bytes_ratio": float(pair["bytes"].sum() / pair["bytes_other"].sum()) if len(pair) else np.nan,
    }
    return table, best, rate, compare


def write_summary(path, table, best, rate, compare, cfg, remaining):
    mode, other = cfg["output_mode"], OTHER_MODE[cfg["output_mode"]]
    lines = [
        f"# Overpass endpoints on {int(table['tiles'].max())} sample tiles ({time.strftime('%Y-%m-%d %H:%M')})",
        "",
        markdown_table(table),
        "",
        f"Query forms ({compare['tiles']} tiles answered in both): '{mode}' finds the same features in "
        f"{compare['same_features']:.0%} of them, takes {compare['seconds_ratio']:.2f} x the time and "
        f"{compare['bytes_ratio']:.2f} x the bytes of '{other}'.",
        "",
        "## Recommended for NEW_02_EXTRACTOSMDATAFULL_GEOJSON.annotations.overpass_endpoints",
        "",
        "```yaml",
        *[f"      - {{url: '{r.url}', workers: {r.workers}}}" for r in best.itertuples()],
        "```",
        "",
        f"Together {rate:.1f} tiles per minute: the {remaining} tiles still without OSM features would take "
        f"about {remaining / rate / 60:.1f} hours."
        if rate
        else "No endpoint met the thresholds.",
    ]
    with atomic_output(path) as tmp, open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Overpass endpoints on sample tiles.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="overpass_benchmark", **overrides)
    bench = cfg["annotations"]
    paths = cfg["paths"]
    # The tiles of NEW_01 around a sample of the plants (the grid file need not exist yet)
    points = gpd.read_file(paths["annotation_points_filepath"])
    points = points[points.geometry.notna() & ~points.geometry.is_empty & (points.geom_type == "Point")]
    check_idx(points, paths["annotation_points_filepath"])
    half = float(bench["cell_size"]) * float(bench["factor"]) / 2
    sample = points.sample(n=min(int(bench["sample_tiles"]), len(points)), random_state=int(bench["random_seed"]))
    sample = square_tiles(sample, half).to_crs(4326)
    sample["bbox"] = sample["geometry"].map(find_bbox)
    tiles = [Tile(int(i), b) for i, b in zip(sample["idx"], sample["bbox"])]
    osm_dir = paths["annotations_by_osm_dir"]
    done = done_tiles(osm_dir) if os.path.isdir(osm_dir) else set()
    remaining = int((~points["idx"].astype(int).isin(done)).sum())

    with ThreadPoolExecutor(max_workers=len(bench["candidates"])) as pool:
        parts = list(pool.map(lambda e: run_endpoint(e, tiles, bench), bench["candidates"]))
    requests = pd.DataFrame([row for part in parts for row in part])
    table, best, rate, compare = summarize(requests, bench)

    out = paths["output_dir"]
    os.makedirs(out, exist_ok=True)
    for name, frame in (("requests.csv", requests), ("endpoints.csv", table)):
        with atomic_output(os.path.join(out, name)) as tmp:
            frame.to_csv(tmp, index=False)
    write_summary(os.path.join(out, "summary.md"), table, best, rate, compare, bench, remaining)
    with atomic_output(os.path.join(out, "results.zip")) as tmp, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("requests.csv", "endpoints.csv", "summary.md"):
            z.write(os.path.join(out, name), name)
    logger.info("\n%s", open(os.path.join(out, "summary.md"), encoding="utf-8").read())


if __name__ == "__main__":
    configure_logging()
    main()
