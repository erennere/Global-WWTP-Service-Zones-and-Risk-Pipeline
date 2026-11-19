"""Render text annotations onto pre-generated imagery tiles.

This script reads per-grid OSM-derived polygon and line features, projects them
into image space, and writes annotated PNG or GeoTIFF outputs. References to
"clip" in this file mean geometric intersection with a tile boundary, not use of
an ML CLIP model.
"""

import os
import argparse
import math
import multiprocessing
import random
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import box
import rasterio
from rasterio.transform import from_origin
from PIL import Image, ImageDraw, ImageFont
from pyproj import Geod, Transformer
import logging

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from .images_ref import image_offsets_m, plant_idx_of_rows
from ..utils import configure_logging, ensure_output_dir_for_file

logger = logging.getLogger(__name__)


# Bundled asset, resolved against the package rather than the working directory.
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "dejavu-sans.book.ttf")
RESOLUTIONS = {
    1: 78271.52,
    2: 39135.76,
    3: 19567.88,
    4: 9783.94,
    5: 4891.97,
    6: 2445.98,
    7: 1222.99,
    8: 611.50,
    9: 305.75,
    10: 152.87,
    11: 76.44,
    12: 38.22,
    13: 19.11,
    14: 9.55,
    15: 4.78,
    16: 2.39,
    17: 1.19,
    18: 0.60,
    19: 0.30,
}
ZOOM_LEVEL = 17
RES_X = RESOLUTIONS[ZOOM_LEVEL]
RES_Y = RESOLUTIONS[ZOOM_LEVEL]
CELL_SIZE = 3072
FACTOR = 1.194
IMAGE_SIZE = [3072, 3072]
BASE_Z17_RES = 1.1943285669555664
MAX_WORKERS = 64
GEOREFERENCED = False
# Label text and its halo, in pixels of the output image (TARGET_SIZE)
LABEL_FONT_PX = 8
LABEL_HALO_PX = 1
DPI = 72
EARTH_RADIUS = 6378137
WORLD_WIDTH = 2 * math.pi * EARTH_RADIUS  # ~40075016.685
TARGET_SIZE = [1024, 1024]
WEB_MERCATOR_TILE_SIZE = 256
IMAGERY_REFERENCE_TILE_SIZE = 512

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
GEOD = Geod(ellps="WGS84")


def get_image(idx, images_dir):
    """Load a pre-generated source image for one annotation index."""
    filepath = os.path.join(images_dir, f"{idx}.png")
    if os.path.exists(filepath):
        return Image.open(filepath)
    else:
        return None


def pixel_size_m():
    """Web Mercator metres per image pixel: BASE_Z17_RES x mercator tile / imagery reference tile (the new
    plants' 3072 px images, 19 Nov 2026: 1.194, so an image shows NEW_01's 3,669 m tile)."""
    return BASE_Z17_RES * (WEB_MERCATOR_TILE_SIZE / IMAGERY_REFERENCE_TILE_SIZE)


def image_frame(x, y):
    """The square an image centred at ``x, y`` (EPSG:3857) shows: IMAGE_SIZE pixels of pixel_size_m()."""
    half = IMAGE_SIZE[0] * pixel_size_m() / 2
    return box(x - half, y - half, x + half, y + half)


def mercator_to_pixel(x, y, cx, cy, IMAGE_SIZE, wrap=True):
    """
    Specifically tuned for Maxar Zoom 17 with a 3072px buffer.
    x, y: Target Web Mercator coordinates (meters)
    cx, cy: Center Web Mercator coordinates of the 3072px image
    """
    # 1. Standard Zoom 17 resolution (meters/pixel for a 256px tile)
    # At Z17, a 256px tile covers ~305.75 meters.
    # At Z17, a 512px tile covers ~611.5 meters.
    # Your 3072px image covers ~3669 meters.

    # 2. Adjust resolution for your specific image size.
    # Since Maxar Zoom 17 typically refers to the 512px tile scale,
    # we use 512 as the divisor if your imagery was pulled as 512px tiles.
    # However, standard Web Mercator math uses 256 as the base unit:
    res = pixel_size_m()

    # 3. Calculate distance from center
    dx = x - cx
    dy = cy - y  # Invert Y: Mercator North is (+), Pixel Down is (+)

    # 4. Handle World Wrap (The "International Date Line" logic)
    if wrap:
        half_world = WORLD_WIDTH / 2
        if dx > half_world:
            dx -= WORLD_WIDTH
        elif dx < -half_world:
            dx += WORLD_WIDTH

    # 5. Map to Pixel Space
    # (Distance / Resolution) + (Center of the 3072px frame)
    px = (dx / res) + (IMAGE_SIZE[0] / 2)
    py = (dy / res) + (IMAGE_SIZE[1] / 2)

    return int(round(px)), int(round(py))


def image_bounds_mercator(center_lon, center_lat):
    """Return approximate Web-Mercator bounds for the configured output image."""
    cx, cy = transformer.transform(center_lon, center_lat)

    initial_res = 2 * math.pi * EARTH_RADIUS / WEB_MERCATOR_TILE_SIZE
    res = initial_res / (2**ZOOM_LEVEL)

    half_w = IMAGE_SIZE[0] * res / 2
    half_h = IMAGE_SIZE[1] * res / 2

    xmin = cx - half_w
    xmax = cx + half_w
    ymin = cy - half_h
    ymax = cy + half_h

    return xmin, ymin, xmax, ymax, res


def draw_text_with_padding(draw, xy, text, font, fill, pad_fill, pad=2):
    """Draws centered text with a stroke/halo outline."""
    draw.text(xy, text, font=font, fill=fill, anchor="mm", stroke_width=pad, stroke_fill=pad_fill)


def draw_rotated_text_with_padding(image, xy, text, angle, font, fill, pad_fill, pad=2):
    """Draws rotated text by compositing a small scratchpad onto the main image."""
    # Create a small scratchpad based on text size
    bbox = font.getbbox(text)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Square buffer to allow rotation without clipping
    pad_dim = int(max(w, h) * 2.0)
    txt_img = Image.new("RGBA", (pad_dim, pad_dim), (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_img)

    center_pt = (pad_dim // 2, pad_dim // 2)
    txt_draw.text(center_pt, text, font=font, fill=fill, anchor="mm", stroke_width=pad, stroke_fill=pad_fill)

    # Note: If labels look 'mirrored', change angle to -angle
    rotated_txt = txt_img.rotate(angle, resample=Image.BICUBIC, expand=False)

    paste_x = int(xy[0] - pad_dim // 2)
    paste_y = int(xy[1] - pad_dim // 2)

    image.alpha_composite(rotated_txt, (paste_x, paste_y))


def linestring_angle(line):
    """Estimate a label rotation angle from the first and last line vertices."""
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[-1]
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def split_grids_for_instance(grids, instance_id, num_instances, split_seed):
    """Deterministically shuffle and split grids into disjoint worker chunks."""
    if num_instances <= 0:
        raise ValueError("num_instances must be > 0")
    if not (0 <= instance_id < num_instances):
        raise ValueError(f"instance_id must be between 0 and {num_instances - 1}, got {instance_id}")

    shuffled = list(grids)
    random.Random(split_seed).shuffle(shuffled)
    return shuffled[instance_id::num_instances]


def draw_annotations(image, annotations, fontsize=12, halo=3):
    """Orchestrates drawing, ensuring lines are drawn before polygon labels (``halo``: outline in pixels)."""
    image = image.convert("RGBA")

    font = ImageFont.truetype(FONT_PATH, fontsize)

    # 1. Draw Rotated Lines First
    #  (Alpha Compositing)
    for ann in [a for a in annotations if a["style"] == "line"]:
        draw_rotated_text_with_padding(
            image, (ann["x"], ann["y"]), ann["text"], ann["angle"], font, "white", "blue", pad=halo
        )

    # 2. Draw Static Labels (Polygons) on top
    draw = ImageDraw.Draw(image)
    for ann in [a for a in annotations if a["style"] != "line"]:
        if ann["style"] == "man_made":
            draw_text_with_padding(draw, (ann["x"], ann["y"]), ann["text"], font, "yellow", "black", pad=halo)
        else:
            draw_text_with_padding(draw, (ann["x"], ann["y"]), ann["text"], font, "black", "white", pad=halo)
    return image


def georef_write(image, center_lon, center_lat, out_path):
    """Write an annotated image as a georeferenced GeoTIFF in EPSG:3857."""
    xmin, ymin, xmax, ymax, res = image_bounds_mercator(center_lon, center_lat)

    transform = from_origin(xmin, ymax, res, res)
    img_arr = np.array(image)

    ensure_output_dir_for_file(out_path)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=img_arr.shape[0],
        width=img_arr.shape[1],
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(img_arr[:, :, i], i + 1)


def process_bbox(idx, bbox_geom, img_idx, poly_gdf, cols, line_gdf, line_cols, output_dir, images_dir):
    """Annotate a single bbox tile with polygon and line labels and write output.

    Parameters
    ----------
    idx : int
        Grid-cell identifier.
    bbox_geom : shapely.geometry.base.BaseGeometry
        Tile geometry in the working CRS.
    img_idx : int
        Source image identifier.
    poly_gdf : geopandas.GeoDataFrame
        Polygon features for the tile.
    cols : list[str]
        Columns to combine into polygon labels.
    line_gdf : geopandas.GeoDataFrame
        Line features for the tile.
    line_cols : list[str]
        Columns to combine into line labels.
    output_dir : str
        Directory where annotated images are written.
    images_dir : str
        Directory containing the source imagery.

    Returns
    -------
    tuple[int, int, str | None]
        Tuple of ``(idx, annotation_count, error_message)``.
    """
    try:
        # 1. Tile Center & Bounds
        center = bbox_geom.centroid
        cx, cy = center.x, center.y

        image = get_image(img_idx, images_dir)
        if image is None:
            return idx, 0, f"Image not found for img_idx={img_idx}"
        annotations = []
        # OSM polygons can be invalid (a ring that crosses itself); GEOS then stops cutting them to the tile
        # with a TopologyException ("side location conflict", cluster 18 Nov). Repaired first; valid ones
        # stay exactly as they are.
        poly_gdf = poly_gdf.set_geometry(poly_gdf.geometry.make_valid())

        # -------- POLYGONS --------
        # Filter first using spatial index (sjoin or intersects) for performance
        possible_polys = poly_gdf[poly_gdf.intersects(bbox_geom)]
        for _, row in possible_polys.iterrows():
            # Filter and collect all valid strings from the specified columns
            valid_tags = [
                str(row[col]) for col in cols if col in row and pd.notna(row[col]) and str(row[col]).lower() != "nan"
            ]
            # Combine them with a comma
            tag = ", ".join(valid_tags) if valid_tags else None
            # Skip if nothing was found
            if not tag:
                continue

            # Geometrically clip the feature to the visible tile footprint.
            clipped_geom = row.geometry.intersection(bbox_geom)
            if clipped_geom.is_empty or not clipped_geom.is_valid:
                continue

            pt = clipped_geom.representative_point()
            px, py = mercator_to_pixel(pt.x, pt.y, cx, cy, IMAGE_SIZE)

            if 0 <= px < IMAGE_SIZE[0] and 0 <= py < IMAGE_SIZE[1]:
                annotations.append(
                    {
                        "x": px,
                        "y": py,
                        "text": tag,
                        "type": "poly",
                        "style": "man_made" if "man_made" in valid_tags else "poly_other",
                        "angle": None,
                    }
                )

        # -------- LINES --------
        possible_lines = line_gdf[line_gdf.intersects(bbox_geom)]
        for _, row in possible_lines.iterrows():
            # Filter and collect all valid strings from the specified columns
            valid_tags = [
                str(row[col]) for col in cols if col in row and pd.notna(row[col]) and str(row[col]).lower() != "nan"
            ]
            # Combine them with a comma
            tag = ", ".join(valid_tags) if valid_tags else None
            # Skip if nothing was found
            if not tag:
                continue

            # Geometric clipping can return MultiLineString fragments when a line
            # enters and leaves the tile multiple times.
            clipped_line = row.geometry.intersection(bbox_geom)
            if clipped_line.is_empty:
                continue

            # Label lines at the midpoint of the visible, geometrically clipped segment.
            pt = clipped_line.interpolate(0.5, normalized=True)
            px, py = mercator_to_pixel(pt.x, pt.y, cx, cy, IMAGE_SIZE)

            # Use the original geometry for angle so label rotation follows the
            # source feature direction rather than a short clipped fragment.
            angle = linestring_angle(row.geometry)
            if 0 <= px < IMAGE_SIZE[0] and 0 <= py < IMAGE_SIZE[1]:
                annotations.append(
                    {
                        "x": px,
                        "y": py,
                        "text": tag,
                        "type": "line",
                        "style": "man_made" if "man_made" in valid_tags else "line",
                        "angle": angle,
                    }
                )

        # -------- DRAW & OUTPUT --------
        # Drawn at the larger of the image and the output size, so the labels stay sharp: an image smaller
        # than the output (512 px -> 1024 px) is enlarged first. Label sizes are given in output pixels.
        draw_px = IMAGE_SIZE[0] if GEOREFERENCED else max(IMAGE_SIZE[0], TARGET_SIZE[0])
        if draw_px != image.size[0]:
            image = image.resize((draw_px, draw_px), resample=Image.LANCZOS)
        scale, per_output_px = draw_px / IMAGE_SIZE[0], draw_px / TARGET_SIZE[0]
        for ann in annotations:
            ann["x"], ann["y"] = int(round(ann["x"] * scale)), int(round(ann["y"] * scale))
        image = draw_annotations(image, annotations, fontsize=max(1, round(LABEL_FONT_PX * per_output_px)),
                                 halo=round(LABEL_HALO_PX * per_output_px))  # fmt: skip
        if GEOREFERENCED:
            out_path = os.path.join(output_dir, f"bbox_{idx}.tif")
            georef_write(image, cx, cy, out_path)
        else:
            out_path = os.path.join(output_dir, f"bbox_{idx}.png")
            if image.size != tuple(TARGET_SIZE):
                image = image.resize(TARGET_SIZE, resample=Image.LANCZOS)
            ensure_output_dir_for_file(out_path)
            image.save(out_path, dpi=(DPI, DPI))
        return idx, len(annotations), None
    except Exception as e:
        logger.exception("bbox %s failed", idx)
        return idx, 0, str(e)


def read_tile_features(folder, grids, kind, columns, files):
    """The ``kind`` features (``polygons`` or ``lines``) of the tiles ``grids`` in EPSG:3857: the tag
    ``columns`` the tiles have, ``geometry`` and the tile number ``grid``.

    Read with GDAL (pyogrio), not DuckDB's ST_READ: ST_READ stops at a tile with tag columns that differ
    only in case (``id`` and ``ID``, ``Acres`` and ``ACRES``: cluster, 18 Nov). ``files``: the names in
    ``folder``; a tile without a file of this kind has no features of it.
    """
    frames = []
    for grid in grids:
        name = f"idx_{grid}_{kind}.geojson"
        if name not in files:
            continue
        gdf = gpd.read_file(os.path.join(folder, name))
        keep = [c for c in columns if c in gdf.columns] + ["geometry"]
        frames.append(gdf[keep].set_crs(4326, allow_override=True).assign(grid=int(grid)))
    if not frames:
        return gpd.GeoDataFrame({"grid": pd.Series(dtype=int)}, geometry=[], crs=3857)
    features = pd.concat(frames, ignore_index=True)
    return features[features.geometry.notna()].to_crs(3857)


# What every tile worker needs besides its task; filled before the pool starts, inherited by fork
_TILES = {}
OSM_OUTPUTS = ("lines.geojson", "polygons.geojson", "empty.txt")


def has_osm_output(idx, osm_files):
    """True when osm_planet wrote the tile ``idx``: its lines, polygons or the marker of an empty tile."""
    return any(f"idx_{int(idx)}_{kind}" in osm_files for kind in OSM_OUTPUTS)


def match_osm_tiles(images, tiles, osm_files, max_offset_m):
    """Which tiles' OSM files hold the features on each image, found by place, not by number.

    ``images``: idx (the plant) and geometry (the image centre, EPSG:3857); ``tiles``: idx and geometry (NEW_01's
    squares, EPSG:3857). Only tiles with OSM output count. Per image (same order): ``osm_tiles`` (tuple of tile
    idx), ``osm_source`` and ``osm_coverage`` (share of the image covered by those tiles):
    - own_tile: the plant's own tile, centred on the image (within ``max_offset_m``);
    - tile_at_image: another tile centred on the image (the tiles were numbered differently);
    - overlapping_tiles: no tile centred on the image; all tiles overlapping it, their features together;
    - none: no tile with OSM output overlaps the image.
    """
    tiles = tiles[[has_osm_output(i, osm_files) for i in tiles["idx"]]].drop_duplicates("idx").reset_index(drop=True)
    tile_idx = tiles["idx"].astype(int).to_numpy()
    centres = tiles.geometry.centroid
    tree = shapely.STRtree(tiles.geometry.to_numpy())
    rows = []
    for idx, point in zip(images["idx"].astype(int), images.geometry):
        frame = image_frame(point.x, point.y)
        near = tree.query(frame, predicate="intersects")
        if not len(near):
            rows.append(((), "none", 0.0))
            continue
        # ground metres: Web Mercator metres x cos(latitude)
        scale = math.cos(math.atan(math.sinh(point.y / EARTH_RADIUS)))
        dist = np.hypot(centres.x.to_numpy()[near] - point.x, centres.y.to_numpy()[near] - point.y) * scale
        own = near[(tile_idx[near] == idx) & (dist <= max_offset_m)]
        if len(own):
            rows.append(((idx,), "own_tile", 1.0))
        elif dist.min() <= max_offset_m:
            rows.append(((int(tile_idx[near[np.argmin(dist)]]),), "tile_at_image", 1.0))
        else:
            covered = shapely.union_all(tiles.geometry.to_numpy()[near]).intersection(frame).area / frame.area
            rows.append((tuple(int(t) for t in tile_idx[near]), "overlapping_tiles", round(float(covered), 4)))
    out = pd.DataFrame(rows, columns=["osm_tiles", "osm_source", "osm_coverage"], index=images.index)
    return out


def read_osm_of_image(osm_tiles, kind, columns):
    """The ``kind`` features of the tiles ``osm_tiles``; a feature in several of them is kept once."""
    t = _TILES
    features = read_tile_features(t["osm_dir"], list(osm_tiles), kind, columns, t["osm_files"])
    if len(osm_tiles) > 1 and len(features):
        key = features["id"].astype(str) if "id" in features.columns else pd.Series("", index=features.index)
        features = features[~(key + features.geometry.to_wkb(hex=True)).duplicated()]
    return features


def annotate_image(idx, img_idx, x, y, osm_tiles):
    """One image from start to end: the OSM features on it, its labels, the annotated image ``bbox_<idx>.png``.
    Returns a report row; a failure is reported, never raised."""
    t = _TILES
    row = {"idx": idx, "image": f"{img_idx}.png", "image_px": None, "osm_features": 0, "osm_features_on_image": 0,
           "labels": 0, "error": None}  # fmt: skip
    frame = image_frame(x, y)
    path = os.path.join(t["images_dir"], row["image"])
    if os.path.exists(path):
        with Image.open(path) as image:  # header only; another size is resized to IMAGE_SIZE before drawing
            row["image_px"] = image.size[0]
    try:
        polys = read_osm_of_image(osm_tiles, "polygons", t["poly_cols"])
        lines = read_osm_of_image(osm_tiles, "lines", t["line_cols"])
        row["osm_features"] = len(polys) + len(lines)
        row["osm_features_on_image"] = int(polys.intersects(frame).sum() + lines.intersects(frame).sum())
    except Exception as e:
        logger.exception("bbox %s: the OSM files of tiles %s could not be read", idx, osm_tiles)
        row["error"] = f"OSM files: {e}"
        return row
    _, n, err = process_bbox(idx, frame, img_idx, polys, t["poly_labels"], lines, t["line_labels"],
                             t["output_dir"], t["images_dir"])  # fmt: skip
    row["labels"], row["error"] = n, err
    return row


def _annotate_image(task):
    return annotate_image(*task)


def run_tasks(func, tasks, workers):
    """``func`` over ``tasks`` in ``workers`` processes (fork: they get _TILES and the configured module
    settings without pickling them), results as they come."""
    if workers > 1 and len(tasks) > 1:
        with multiprocessing.get_context("fork").Pool(min(workers, len(tasks))) as pool:
            yield from pool.imap_unordered(func, tasks)
    else:
        yield from map(func, tasks)


def annotate_images(images, workers, **shared):
    """Annotate ``images`` (columns idx, img_idx, osm_tiles, geometry = image centre in EPSG:3857) in
    ``workers`` processes, one image per task. Never stops for a single image: returns one report row per
    image (OSM features, those on the image, labels, error).

    ``shared``: ``osm_dir``, ``osm_files`` (the names in it), ``poly_cols``/``line_cols`` (tag columns read),
    ``poly_labels``/``line_labels`` (columns drawn), ``output_dir``, ``images_dir``.
    """
    missing_cols = {"idx", "img_idx", "osm_tiles", "geometry"}.difference(images.columns)
    if missing_cols:
        raise KeyError(f"images is missing required columns: {sorted(missing_cols)}")
    tasks = [(int(i), int(img), float(g.x), float(g.y), tuple(o))
             for i, img, g, o in zip(images["idx"], images["img_idx"], images.geometry, images["osm_tiles"])]  # fmt: skip
    _TILES.clear()
    _TILES.update(shared)
    logger.info("Queued %s images for annotation in %s processes", len(tasks), workers)
    rows = []
    for row in run_tasks(_annotate_image, tasks, workers):
        if row["error"]:
            logger.error("bbox %s failed: %s", row["idx"], row["error"])
        else:
            logger.info("bbox %s done (%s tags)", row["idx"], row["labels"])
        rows.append(row)
    return pd.DataFrame(
        rows, columns=["idx", "image", "image_px", "osm_features", "osm_features_on_image", "labels", "error"]
    )


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Annotate Bing images for a deterministic subset of grids.")
    parser.add_argument(
        "instance_id",
        type=int,
        help="Worker index in [0, num_instances-1]. For 10 workers use 0..9.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=10,
        help="Total parallel script instances (default: 10).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for the deterministic random grid split (default: config annotations.random_seed).",
    )
    add_standard_override_arguments(parser)
    args = parser.parse_args()

    logger.info("Starting Bing annotation pipeline")
    try:
        overrides = parse_config_overrides(args=args)
    except ValueError as exc:
        parser.error(str(exc))

    cfg = load_config(script_name="download_bing_annotate", **overrides)
    logger.info("Configuration loaded")

    annotations_cfg = cfg["annotations"]
    if args.split_seed is None:
        args.split_seed = int(annotations_cfg["random_seed"])
    CELL_SIZE = int(annotations_cfg["cell_size"])
    FACTOR = float(annotations_cfg["factor"])
    image_size_px = int(annotations_cfg["image_size_px"])
    IMAGE_SIZE = [image_size_px, image_size_px]
    ZOOM_LEVEL = int(annotations_cfg["zoom_level"])
    RES_X = RESOLUTIONS[ZOOM_LEVEL]
    RES_Y = RESOLUTIONS[ZOOM_LEVEL]
    BASE_Z17_RES = float(annotations_cfg["base_z17_resolution"])
    MAX_WORKERS = int(annotations_cfg["max_workers"])
    GEOREFERENCED = bool(annotations_cfg["georeferenced"])
    LABEL_FONT_PX = float(annotations_cfg["label_font_px"])
    LABEL_HALO_PX = float(annotations_cfg["label_halo_px"])
    DPI = int(annotations_cfg["dpi"])
    target_size_px = int(annotations_cfg["target_size_px"])
    TARGET_SIZE = [target_size_px, target_size_px]
    WEB_MERCATOR_TILE_SIZE = int(annotations_cfg["mercator_tile_size_px"])
    IMAGERY_REFERENCE_TILE_SIZE = int(annotations_cfg["imagery_reference_tile_size_px"])
    EARTH_RADIUS = float(annotations_cfg["earth_radius_m"])
    WORLD_WIDTH = 2 * math.pi * EARTH_RADIUS

    images_dir = cfg["paths"]["annotations_images_dir"]
    grid_filedir = cfg["paths"]["annotations_grid_dir"]
    geojson_file_dir = cfg["paths"]["annotations_by_osm_dir"]

    points_path = f"{images_dir}/ref.geojson"
    grids_filepath = os.path.join(grid_filedir, f"grids_{os.path.basename(cfg['paths']['annotation_points_filepath'])}")

    max_offset_m = float(cfg["max_image_offset_m"])
    images_tag = "-".join(os.path.normpath(images_dir).split(os.sep)[-2:])
    tiles = gpd.read_file(grids_filepath).to_crs(3857)
    tiles = tiles[tiles.geometry.notna() & ~tiles.geometry.is_empty].copy()
    tiles["idx"] = tiles["idx"].astype(int)
    osm_files = set(os.listdir(geojson_file_dir))

    # The images: which plant each one shows, and which file is its image. Nothing here stops the run; what
    # does not fit is fixed where it can be and written to the match report.
    ref = gpd.read_file(points_path).to_crs(3857)
    ref["ref_row"] = np.arange(len(ref))
    list_idx = pd.read_csv(cfg["paths"]["images_ref_list"], usecols=["idx"])["idx"]
    inside = ref["idx"].astype(int).between(0, len(list_idx) - 1)
    images = plant_idx_of_rows(ref[inside], cfg["paths"]["images_ref_list"])
    # <n>.png: the image of the plant whose ref.geojson idx is n (new plants' set, checked by eye on 4 images),
    # or of ref.geojson's n-th feature
    images["img_idx"] = images["list_row" if annotations_cfg["image_names"] == "ref_idx" else "ref_row"]
    plants = gpd.read_file(cfg["paths"]["annotation_points_filepath"], columns=["idx"])
    plants["idx"] = plants["idx"].astype(int)
    images["image_to_plant_m"] = image_offsets_m(images, plants).round(2)
    # The OSM features of each image: from the tiles at its place (its own tile when that is centred on it)
    images = images.join(match_osm_tiles(images, tiles, osm_files, max_offset_m))
    logger.info(
        "%s images (%s with a ref.geojson idx outside %s); %s farther than %g m from their plant; OSM from: %s",
        len(images), int((~inside).sum()), cfg["paths"]["images_ref_list"],
        int((images["image_to_plant_m"].fillna(np.inf) > max_offset_m).sum()), max_offset_m,
        images["osm_source"].value_counts().to_dict(),
    )  # fmt: skip

    output_dir = cfg["paths"]["annotated_images_output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    plant_numbers = split_grids_for_instance(
        images["idx"].unique().tolist(),
        instance_id=args.instance_id,
        num_instances=args.num_instances,
        split_seed=args.split_seed,
    )
    mine = images[images["idx"].isin(plant_numbers)]
    logger.info("Instance %s/%s will draw %s images (seed=%s)", args.instance_id, args.num_instances, len(mine),
                args.split_seed)  # fmt: skip
    drawn = annotate_images(
        mine,
        MAX_WORKERS,
        osm_dir=geojson_file_dir,
        osm_files=osm_files,
        poly_cols=["id", "man_made", "landuse", "industrial", "power", "resource"],
        line_cols=["waterway", "man_made", "landuse", "industrial", "power", "resource", "water", "id"],
        poly_labels=["man_made", "landuse", "industrial", "power", "resource", "water"],
        line_labels=["waterway", "man_made", "landuse", "industrial", "power", "resource", "water"],
        output_dir=output_dir,
        images_dir=images_dir,
    )

    # The match report of this share: one row per image (and, in share 0, the ref.geojson rows outside the list)
    columns = ["ref_row", "list_row", "img_idx", "plant_id", "ISO_2", "image_to_plant_m", "osm_source", "osm_tiles",
               "osm_coverage"]  # fmt: skip
    report = mine.assign(image=mine["img_idx"].astype(int).astype(str) + ".png")
    report = report[["idx", "image", *[c for c in columns if c in report.columns]]].merge(drawn, on=["idx", "image"],
                                                                                         how="left")  # fmt: skip
    if args.instance_id == 0 and (~inside).any():
        outside = pd.DataFrame({"ref_row": ref.loc[~inside, "ref_row"], "list_row": ref.loc[~inside, "idx"],
                                "error": "ref.geojson idx outside images_ref_list"})  # fmt: skip
        report = pd.concat([report, outside], ignore_index=True)
    report["osm_tiles"] = report["osm_tiles"].map(lambda t: " ".join(map(str, t)) if isinstance(t, tuple) else t)
    report_path = os.path.join(cfg["paths"]["match_report_dir"], f"{images_tag}_shard{args.instance_id}.csv")
    ensure_output_dir_for_file(report_path)
    report.to_csv(report_path, index=False)
    failed = report["error"].notna()
    logger.info(
        "Annotation finished: %s images drawn, %s failed, %s without OSM features on them; match report %s",
        int((~failed).sum()), int(failed.sum()), int((report["osm_features_on_image"].fillna(0) == 0).sum()),
        report_path,
    )  # fmt: skip
