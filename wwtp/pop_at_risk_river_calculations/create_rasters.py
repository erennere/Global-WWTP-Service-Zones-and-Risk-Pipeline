"""Non-served population per country and the islands it forms, from the WorldPop rasters of one year.

For every country raster of ``population_year``:

1. Country totals: people inside the country's service areas (served) and outside them (not served). A
   pixel is served when its centre lies in a service area of the same country (``served_all_touched``: any
   part of it). Only the country's own service areas count, as add_pop counts each plant's people from its
   own country's raster.
2. Islands: 8-connected populated, non-served pixels of one HydroBASINS basin. An island is kept when it has
   at least ``min_pixels`` pixels and more than ``threshold_value`` people, the whole island (Eren, 24 Nov
   2026; before, its pieces per zoom-8 tile were tested one by one in find_unserved_pop). The basins are the
   ones the raster covers, by location; before, a basin was only processed for the one country it overlaps
   most, so its populated pixels in the neighbouring country were never looked at.

How:
  * One pass over the raster in bands of 512 rows: the served mask is rasterised once per band, the sums are
    taken, and a 512 x 512 block grid notes where people live. A raster stored in strips (a few rows per
    block) and wider than 8,192 pixels is copied to a tiled file in the temp folder during this pass, so that
    reading one basin does not decompress whole rows of the country.
  * One task per basin that holds people: read its window once, rasterise the basin and the service areas
    once, label the islands (scipy.ndimage.label), sum people and pixels per island from the pixels
    (np.bincount), and turn only the kept islands into polygons.
  * Countries go to shards largest first (by file size), and within a shard the basin tasks of all countries
    share one process pool, largest first.
  Before: a loop over every raster block with one mask per basin per block, a polygon for every non-served
  pixel cluster, unions per basin, a second read of the raster per island (exact_extract), and a signed
  GeoTIFF per country that nothing read.

Outputs (CSV, appended per country when it is complete; shards write their own and are merged):
  csv_output_filepath  country, population_served, population_unserved, population_total, population_served_index
  non_served_outpath   one row per island: geometry (WKT, EPSG:4326), the basin's columns (basin_column_name,
                       NEXT_DOWN, NEXT_SINK, MAIN_BAS), pop_sum, pixel_count, country

``--countries LU,NL --out-dir DIR`` runs only these countries and writes the two CSVs to DIR (a check run).
"""

import argparse
import gc
import logging
import math
import os
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio import windows
from rasterio.features import rasterize, shapes
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import box, shape
from shapely.ops import unary_union

from .. import provenance
from ..add_pop import find_country_tif_files_for_year
from ..geo_utils import repair_geometry
from ..pipelines import create_pop_output_paths
from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..utils import atomic_output, configure_logging, ensure_output_dir_for_file

logger = logging.getLogger(__name__)

BLOCK = 512  # rows per band of the country pass, block size of the occupancy grid and of the tiled copy
STRIP_MAX_ROWS = 64  # a source whose blocks have fewer rows than this is stored in strips
COPY_MIN_WIDTH = 8192  # narrower striped rasters are read as they are
BASIN_META = ("NEXT_DOWN", "NEXT_SINK", "MAIN_BAS")
EIGHT = np.ones((3, 3), dtype=bool)  # 8-connectivity


def _sanitize_polygon_geom(geom):
    """Return a valid Polygon/MultiPolygon geometry or None."""
    geom = repair_geometry(geom)
    if geom is None:
        return None

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom

    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if (not g.is_empty) and g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        merged = unary_union(polys)
        if merged.is_empty:
            return None
        if merged.geom_type in ("Polygon", "MultiPolygon"):
            return merged

    return None


def geotiff_exists_and_valid(path):
    """Return True when a GeoTIFF exists and raster metadata can be read."""
    if not os.path.exists(path):
        return False
    try:
        with rasterio.open(path) as src:
            _ = src.count  # forces metadata read
        return True
    except Exception:
        return False


def _clean(block, nodata):
    """People per pixel as float32: nodata, NaN and values <= 0 become 0."""
    block = np.asarray(block, dtype=np.float32)
    if nodata is not None and not np.isnan(nodata):
        block[block == np.float32(nodata)] = 0
    block[~(block > 0)] = 0
    return block


def _mask(geoms, out_shape, transform, all_touched=False):
    """Boolean raster of the pixels the geometries cover (centre rule unless all_touched)."""
    if len(geoms) == 0:
        return np.zeros(out_shape, dtype=bool)
    return rasterize(
        ((g, 1) for g in geoms), out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8,
        all_touched=all_touched,
    ).view(bool)


def _is_striped(src):
    return src.block_shapes[0][0] < STRIP_MAX_ROWS and src.width > COPY_MIN_WIDTH


def country_pass(raster_path, served_wkb, all_touched=False, copy_dir=None):
    """Served / not-served sums of one country raster, the blocks that hold people, and the file to read basins from.

    Returns a dict: served, unserved, occupied (bool grid, one cell per 512 x 512 block), read_path (a tiled
    copy in ``copy_dir`` when the source is stored in strips, else the source), crs, transform, height, width.
    """
    served_geoms = shapely.from_wkb(np.asarray(served_wkb, dtype=object)) if len(served_wkb) else np.array([])
    tree = shapely.STRtree(served_geoms) if len(served_geoms) else None
    with rasterio.open(raster_path) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f"{raster_path}: expected a WorldPop raster in EPSG:4326, got {src.crs}")
        height, width, transform = src.height, src.width, src.transform
        occupied = np.zeros((math.ceil(height / BLOCK), math.ceil(width / BLOCK)), dtype=bool)
        read_path, dst = raster_path, None
        if copy_dir is not None and _is_striped(src):
            read_path = os.path.join(copy_dir, f"tiled_{os.path.basename(raster_path)}")
            profile = dict(
                driver="GTiff", width=width, height=height, count=1, dtype="float32", crs=src.crs,
                transform=transform, nodata=None, tiled=True, blockxsize=BLOCK, blockysize=BLOCK,
                compress="deflate", predictor=3, BIGTIFF="IF_SAFER",
            )
            try:
                dst = rasterio.open(read_path, "w", **profile)
            except Exception as err:  # no room in the temp folder: read the source as it is
                logger.warning("No tiled copy of %s (%s); basins read the source", raster_path, err)
                read_path, dst = raster_path, None
        served = unserved = 0.0
        try:
            for r0 in range(0, height, BLOCK):
                win = Window(0, r0, width, min(BLOCK, height - r0))
                data = _clean(src.read(1, window=win), src.nodata)
                if dst is not None:
                    try:
                        dst.write(data, 1, window=win)
                    except Exception as err:
                        logger.warning("Tiled copy of %s failed (%s); basins read the source", raster_path, err)
                        dst.close()
                        os.remove(read_path)
                        read_path, dst = raster_path, None
                populated = data > 0
                if not populated.any():
                    continue
                cols = np.add.reduceat(populated.any(axis=0), np.arange(0, width, BLOCK)) > 0
                occupied[r0 // BLOCK] |= cols
                total = float(data.sum(dtype=np.float64))
                inside = 0.0
                if tree is not None:
                    hits = tree.query(box(*windows.bounds(win, transform)))
                    if len(hits):
                        m = _mask(served_geoms[hits], data.shape, windows.transform(win, transform), all_touched)
                        inside = float(np.sum(data, where=m, dtype=np.float64))
                served += inside
                unserved += total - inside
        finally:
            if dst is not None:
                dst.close()
        return dict(
            served=served, unserved=unserved, occupied=occupied, read_path=read_path, crs=src.crs,
            transform=transform, height=height, width=width,
        )


def basin_window(bounds, transform, height, width):
    """The pixel window of ``bounds`` (whole pixels, clipped to the raster), or None outside it."""
    w = windows.from_bounds(*bounds, transform=transform)
    c0, c1 = max(0, math.floor(w.col_off)), min(width, math.ceil(w.col_off + w.width))
    r0, r1 = max(0, math.floor(w.row_off)), min(height, math.ceil(w.row_off + w.height))
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


def basin_islands(read_path, window, basin_wkb, basin_attrs, served_wkb, min_pixels, min_pop, all_touched=False):
    """The kept islands of one basin: rows with geometry (shapely, raster CRS), pop_sum, pixel_count and the basin columns."""
    with rasterio.open(read_path) as src:
        data = _clean(src.read(1, window=window), src.nodata)
        wt = windows.transform(window, src.transform)
    mask = data > 0
    if not mask.any():
        return []
    # pixel centres in the basin; HydroBASINS edges lie on its 15" grid, never through a WorldPop (3") pixel
    # centre, so every pixel belongs to exactly one basin whatever window it is read in
    mask &= _mask([shapely.from_wkb(basin_wkb)], data.shape, wt)
    if len(served_wkb) and mask.any():
        mask &= ~_mask(shapely.from_wkb(np.asarray(served_wkb, dtype=object)), data.shape, wt, all_touched)
    if not mask.any():
        return []
    labels, n = ndimage.label(mask, structure=EIGHT)
    lab = labels[mask]
    pop = np.round(np.bincount(lab, weights=data[mask], minlength=n + 1)).astype(np.int64)
    px = np.bincount(lab, minlength=n + 1).astype(np.int64)
    keep = (px >= min_pixels) & (pop > min_pop)
    keep[0] = False
    if not keep.any():
        return []
    del data, lab
    parts = {}
    for geom, value in shapes(labels, mask=keep[labels], connectivity=8, transform=wt):
        parts.setdefault(int(value), []).append(shape(geom))
    rows = []
    for value, geoms in parts.items():
        geom = _sanitize_polygon_geom(geoms[0] if len(geoms) == 1 else unary_union(geoms))
        if geom is not None:
            rows.append({**basin_attrs, "geometry": geom, "pop_sum": int(pop[value]), "pixel_count": int(px[value])})
    return rows


def _append_csv(df, filepath):
    """Append ``df`` to ``filepath``, writing the header only on first creation."""
    ensure_output_dir_for_file(filepath)
    exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    df.to_csv(filepath, index=False, mode="a" if exists else "w", header=not exists)


def _finished_countries(csv_output_filepath, non_served_outpath):
    """Countries already in the stats CSV; island rows of any other country (an interrupted run) are removed."""
    if not (os.path.exists(csv_output_filepath) and os.path.getsize(csv_output_filepath) > 0):
        done = set()
    else:
        done = set(pd.read_csv(csv_output_filepath, keep_default_na=False, na_values=[""]).country.astype(str))
    if os.path.exists(non_served_outpath) and os.path.getsize(non_served_outpath) > 0:
        islands = pd.read_csv(non_served_outpath, keep_default_na=False, na_values=[""])
        stale = ~islands.country.astype(str).isin(done)
        if stale.any():
            logger.warning("Removing %s island rows of unfinished countries from %s", int(stale.sum()), non_served_outpath)
            with atomic_output(non_served_outpath) as tmp_path:
                islands[~stale].to_csv(tmp_path, index=False)
    return done


def process_countries(
    tif_dict,
    served_gdf,
    basins_gdf,
    csv_output_filepath,
    non_served_outpath,
    max_workers=4,
    min_pixels=9,
    min_pop=0,
    country_col="ISO_2",
    basin_col="HYBAS_ID",
    all_touched=False,
    copy_dir=None,
):
    """Country totals and islands of every country in ``tif_dict``; a country's rows are written when it is complete.

    Countries already in the stats CSV are skipped (a rerun continues). Raises at the end if a country failed,
    after writing every country that did not.
    """
    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")
    done = _finished_countries(csv_output_filepath, non_served_outpath)
    countries = sorted((c for c in tif_dict if c not in done), key=lambda c: (-_size(tif_dict[c]), c))
    if done:
        logger.info("Skipping %s countries finished before", len([c for c in tif_dict if c in done]))
    if not countries:
        return {}
    basins_gdf = basins_gdf.reset_index(drop=True)
    basin_tree = shapely.STRtree(basins_gdf.geometry.values)
    meta_cols = [c for c in (basin_col, *BASIN_META) if c in basins_gdf.columns]
    served_by_country = {c: np.asarray(g.geometry.values, dtype=object) for c, g in served_gdf.groupby(country_col)}
    copy_dir = copy_dir or tempfile.mkdtemp(prefix="create_rasters_")
    state, results = {}, {}

    def finish(country):
        st = state.pop(country)
        rows = st["rows"]
        islands = gpd.GeoDataFrame(rows, geometry="geometry", crs=st["crs"]) if rows else None
        if islands is not None:
            table = pd.DataFrame(islands.drop(columns="geometry"))
            table["country"] = country
            table["geometry"] = islands.geometry.to_wkt().values
            _append_csv(table, non_served_outpath)
        s, u = st["served"], st["unserved"]
        _append_csv(
            pd.DataFrame(
                {
                    "country": [country],
                    "population_served": [s],
                    "population_unserved": [u],
                    "population_total": [s + u],
                    "population_served_index": [s / (s + u + 0.1)],
                }
            ),
            csv_output_filepath,
        )
        if st["read_path"] != tif_dict[country] and os.path.exists(st["read_path"]):
            os.remove(st["read_path"])
        logger.info(
            "[%s] done in %.0f s: %s basins, %s islands, %.0f served, %.0f not served",
            country, time.time() - st["t0"], st["n_basins"], len(rows), s, u,
        )
        results[country] = True

    with ProcessPoolExecutor(max_workers=int(max_workers)) as executor:
        pending = {}
        for country in countries:
            served = served_by_country.get(country, np.array([], dtype=object))
            fut = executor.submit(country_pass, tif_dict[country], shapely.to_wkb(served), all_touched, copy_dir)
            pending[fut] = ("pass", country)
            state[country] = {"t0": time.time(), "served_geoms": served}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                kind, country = pending.pop(fut)
                if country not in state:  # failed already
                    continue
                st = state[country]
                try:
                    out = fut.result()
                except Exception as err:
                    logger.exception("[FAIL] %s (%s): %s", country, kind, err)
                    results[country] = False
                    if os.path.exists(st.get("read_path", "")) and st["read_path"] != tif_dict[country]:
                        os.remove(st["read_path"])
                    state.pop(country)
                    continue
                if kind == "basin":
                    st["rows"].extend(out)
                    st["left"] -= 1
                    if st["left"] == 0:
                        finish(country)
                    continue
                # the country pass is back: one task per basin that holds people
                st.update(out, rows=[])
                tasks = _basin_tasks(out, basins_gdf, basin_tree, meta_cols, st.pop("served_geoms"))
                st["n_basins"] = st["left"] = len(tasks)
                logger.info("[%s] pass done in %.0f s; %s basins with people", country, time.time() - st["t0"], len(tasks))
                if not tasks:
                    finish(country)
                for window, basin_wkb, attrs, served_wkb in tasks:
                    f = executor.submit(
                        basin_islands, out["read_path"], window, basin_wkb, attrs, served_wkb, min_pixels, min_pop,
                        all_touched,
                    )
                    pending[f] = ("basin", country)
            gc.collect()
    shutil.rmtree(copy_dir, ignore_errors=True)
    failed = sorted(c for c, ok in results.items() if not ok)
    if failed:
        raise RuntimeError(f"create_rasters failed for {len(failed)} countries: {', '.join(failed)}")
    return results


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _basin_tasks(country_out, basins_gdf, basin_tree, meta_cols, served_geoms):
    """(window, basin WKB, basin columns, served WKB) for every basin whose window touches a block with people, largest first."""
    t, h, w, occ = country_out["transform"], country_out["height"], country_out["width"], country_out["occupied"]
    bounds = rasterio.transform.array_bounds(h, w, t)  # (west, south, east, north)
    served_tree = shapely.STRtree(served_geoms) if len(served_geoms) else None
    tasks = []
    for i in basin_tree.query(box(bounds[0], bounds[1], bounds[2], bounds[3])):
        geom = basins_gdf.geometry.values[i]
        win = basin_window(geom.bounds, t, h, w)
        if win is None:
            continue
        r0, c0 = win.row_off // BLOCK, win.col_off // BLOCK
        r1, c1 = (win.row_off + win.height - 1) // BLOCK + 1, (win.col_off + win.width - 1) // BLOCK + 1
        if not occ[r0:r1, c0:c1].any():
            continue
        attrs = {c: basins_gdf[c].values[i] for c in meta_cols}
        attrs = {c: (v.item() if hasattr(v, "item") else v) for c, v in attrs.items()}
        served = served_geoms[served_tree.query(box(*geom.bounds))] if served_tree is not None else []
        tasks.append((win, shapely.to_wkb(geom), attrs, shapely.to_wkb(np.asarray(served, dtype=object))))
    tasks.sort(key=lambda task: -(task[0].width * task[0].height))
    return tasks


def parse_args(argv=None):
    """Parse sharding args and standardized named config overrides."""
    parser = argparse.ArgumentParser(description="Country totals and non-served islands of a country shard.")
    parser.add_argument(
        "--job-index", type=int, default=0, help="Index of this shard, in [0, total-jobs-1] (default: 0)."
    )
    parser.add_argument(
        "--total-jobs", type=int, default=1, help="Total number of shards the country list is split into (default: 1)."
    )
    parser.add_argument(
        "--merge-shards",
        type=int,
        default=None,
        metavar="N",
        help="Only concatenate the CSVs of shards 0..N-1 into the final CSVs and exit.",
    )
    parser.add_argument("--countries", default=None, help="Check run: only these ISO-2 codes (comma-separated).")
    parser.add_argument("--out-dir", default=None, help="Check run: write the two CSVs here instead.")
    add_standard_override_arguments(parser)
    return parser.parse_args(argv)


def shard_path(path, job_index, total_jobs):
    """Where shard ``job_index`` of ``total_jobs`` writes the CSV ``path``; ``path`` itself for a single shard.

    Shards never append to a shared file (two array tasks used to write
    interleaved rows and two headers); ``merge_shards`` concatenates them.
    """
    if total_jobs == 1:
        return path
    stem, ext = os.path.splitext(path)
    return f"{stem}_shard{job_index}of{total_jobs}{ext}"


def merge_shards(paths, total_jobs):
    """Concatenate the shard CSVs of each of ``paths`` into ``path`` (empty shards contribute nothing)."""
    for path in paths:
        parts = [shard_path(path, i, total_jobs) for i in range(total_jobs)]
        missing = [part for part in parts if not os.path.exists(part)]
        if missing:
            raise FileNotFoundError(f"shards not finished: {missing}")
        frames = [pd.read_csv(part, keep_default_na=False, na_values=[""]) for part in parts if os.path.getsize(part) > 0]
        with atomic_output(path) as tmp_path:
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(tmp_path, index=False)
            else:
                open(tmp_path, "w").close()
        logger.info("Merged %s shards into %s", total_jobs, path)


def shard_tif_dict(tif_dict, job_index, total_jobs, seed=None):
    """The countries of shard ``job_index``: largest raster files first, each to the shard with the least so far.

    Balances the shards by file size (before: a random split, so one shard could get Russia, China and the
    United States). ``seed`` is not used any more; kept for the call signature.
    """
    if total_jobs < 1:
        raise ValueError(f"total_jobs must be >= 1, got {total_jobs}")
    if job_index < 0 or job_index >= total_jobs:
        raise ValueError(f"job_index must be in [0, {total_jobs - 1}], got {job_index}")
    load = [(0, 0)] * total_jobs  # (bytes, countries) per shard
    mine = {}
    for country in sorted(tif_dict, key=lambda c: (-_size(tif_dict[c]), c)):
        shard = min(range(total_jobs), key=lambda i: (load[i], i))
        load[shard] = (load[shard][0] + _size(tif_dict[country]), load[shard][1] + 1)
        if shard == job_index:
            mine[country] = tif_dict[country]
    return mine


def main(argv=None):
    """Load the configuration, the service areas and the basins, and run the shard (or a check run)."""
    args = parse_args(argv)
    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name="create_rasters", **overrides)
    provenance.write_run_manifest(cfg, "create_rasters")
    non_served_outpath = os.path.abspath(cfg["paths"]["non_served_outpath"].replace(".gpkg", ".csv"))
    csv_output_filepath = os.path.abspath(cfg["paths"]["csv_output_filepath"].replace(".gpkg", ".csv"))
    if args.merge_shards is not None:
        merge_shards([csv_output_filepath, non_served_outpath], args.merge_shards)
        return
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        csv_output_filepath = os.path.join(os.path.abspath(args.out_dir), os.path.basename(csv_output_filepath))
        non_served_outpath = os.path.join(os.path.abspath(args.out_dir), os.path.basename(non_served_outpath))
    else:
        non_served_outpath = shard_path(non_served_outpath, args.job_index, args.total_jobs)
        csv_output_filepath = shard_path(csv_output_filepath, args.job_index, args.total_jobs)

    approach = str(cfg["figures"]["approach"])
    voronoi_map = create_pop_output_paths(cfg)["voronoi"]
    if approach not in voronoi_map:
        raise KeyError(
            f"Configured figures.approach '{approach}' is not available; expected one of {sorted(voronoi_map)}"
        )
    country_col, basin_col = cfg["country_output_column"], cfg["basin_column_name"]
    layer = os.path.abspath(voronoi_map[approach])
    logger.info("Loading service areas from %s", layer)
    served = gpd.read_file(layer, columns=[country_col])
    if served.crs is None:
        raise ValueError(f"{layer} has no CRS")
    served = served[served.geometry.notna() & ~served.geometry.is_empty].to_crs(4326)

    year = int(cfg["population_year"])
    tif_dict = find_country_tif_files_for_year(served[country_col].dropna().unique(), cfg["paths"]["pop_tif_dir"], year)
    if args.countries:
        wanted = {c.strip().upper() for c in args.countries.split(",") if c.strip()}
        tif_dict = {c: p for c, p in tif_dict.items() if c in wanted}
    else:
        tif_dict = shard_tif_dict(tif_dict, args.job_index, args.total_jobs)
    logger.info("%s countries with a %s raster in this run (shard %s/%s)", len(tif_dict), year, args.job_index, args.total_jobs)

    basins = gpd.read_file(cfg["paths"]["watershed"])
    if basins.crs is None:
        raise ValueError("Watershed dataset must include CRS metadata")
    keep = [c for c in (basin_col, *BASIN_META) if c in basins.columns]
    basins = (
        basins[[*keep, "geometry"]].to_crs(4326).drop_duplicates(subset=[basin_col, "geometry"]).reset_index(drop=True)
    )

    process_countries(
        tif_dict,
        served,
        basins,
        csv_output_filepath,
        non_served_outpath,
        max_workers=int(cfg["annotations"]["max_workers"]),
        min_pixels=int(cfg["min_pixels"]),
        min_pop=int(cfg["threshold_value"]),
        country_col=country_col,
        basin_col=basin_col,
        all_touched=bool(cfg["served_all_touched"]),
    )
    if args.total_jobs > 1 and not args.out_dir:
        for path in (csv_output_filepath, non_served_outpath):  # a shard without rows still reports that it finished
            if not os.path.exists(path):
                ensure_output_dir_for_file(path)
                open(path, "w").close()


if __name__ == "__main__":
    configure_logging()
    main()
