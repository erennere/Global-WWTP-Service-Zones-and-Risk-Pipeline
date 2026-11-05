"""Render text annotations onto pre-generated imagery tiles.

This script reads per-grid OSM-derived polygon and line features, projects them
into image space, and writes annotated PNG or GeoTIFF outputs. References to
"clip" in this file mean geometric intersection with a tile boundary, not use of
an ML CLIP model.
"""

import os, sys
import argparse
import math
import random
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyproj import Transformer
import logging
import shapely.wkt

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..geo_utils import ensure_duckdb_spatial
from ..utils import (
    configure_logging,
    duckdb_connection,
    ensure_output_dir_for_file,
)

logger = logging.getLogger(__name__)



def safe_wkt_load(wkt_wtr):
    """Parse WKT text into a Shapely geometry, returning None on bad rows."""
    try:
        if not wkt_wtr or not isinstance(wkt_wtr, str):
            return None
        return shapely.wkt.loads(wkt_wtr)
    except Exception:
        return None
    
# Bundled asset, resolved against the package rather than the working directory.
FONT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "dejavu-sans.book.ttf"
)
RESOLUTIONS = { 
    1 :	78271.52,   
    2 :	39135.76,	
    3 :	19567.88,	
    4 :	9783.94,	
    5 : 4891.97,	
    6 :	2445.98,	
    7 :	1222.99,	
    8 :	611.50,	
    9 :	305.75,	
    10 : 152.87,	
    11 : 76.44,	
    12 : 38.22,
    13 : 19.11,
    14 : 9.55,
    15 : 4.78,
    16 : 2.39,
    17 : 1.19,
    18 : 0.60,
    19 : 0.30
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
FONTSIZE = 24
DPI = 72 
EARTH_RADIUS = 6378137
WORLD_WIDTH = 2 * math.pi * EARTH_RADIUS  # ~40075016.685
TARGET_SIZE = [1024, 1024]
WEB_MERCATOR_TILE_SIZE = 256
IMAGERY_REFERENCE_TILE_SIZE = 512

transformer = Transformer.from_crs(
    "EPSG:4326", "EPSG:3857", always_xy=True
)

def get_image(idx, images_dir):
    """Load a pre-generated source image for one annotation index."""
    filepath = os.path.join(images_dir, f'{idx}.png')
    if os.path.exists(filepath):
        return Image.open(filepath)
    else:
        return None

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
    res = BASE_Z17_RES * (WEB_MERCATOR_TILE_SIZE / IMAGERY_REFERENCE_TILE_SIZE)
    
    # 3. Calculate distance from center
    dx = x - cx
    dy = cy - y  # Invert Y: Mercator North is (+), Pixel Down is (+)

    # 4. Handle World Wrap (The "International Date Line" logic)
    if wrap:
        half_world = WORLD_WIDTH / 2
        if dx > half_world: dx -= WORLD_WIDTH
        elif dx < -half_world: dx += WORLD_WIDTH

    # 5. Map to Pixel Space
    # (Distance / Resolution) + (Center of the 3072px frame)
    px = (dx / res) + (IMAGE_SIZE[0] / 2)
    py = (dy / res) + (IMAGE_SIZE[1] / 2)

    return int(round(px)), int(round(py))

def image_bounds_mercator(center_lon, center_lat):
    """Return approximate Web-Mercator bounds for the configured output image."""
    cx, cy = transformer.transform(center_lon, center_lat)

    initial_res = 2 * math.pi * EARTH_RADIUS / WEB_MERCATOR_TILE_SIZE
    res = initial_res / (2 ** ZOOM_LEVEL)

    half_w = IMAGE_SIZE[0] * res / 2
    half_h = IMAGE_SIZE[1] * res / 2

    xmin = cx - half_w
    xmax = cx + half_w
    ymin = cy - half_h
    ymax = cy + half_h

    return xmin, ymin, xmax, ymax, res

def draw_text_with_padding(draw, xy, text, font, fill, pad_fill, pad=2):
    """Draws centered text with a stroke/halo outline."""
    draw.text(
        xy, text, font=font, fill=fill,
        anchor="mm", stroke_width=pad, stroke_fill=pad_fill
    )

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
    txt_draw.text(
        center_pt, text, font=font, fill=fill,
        anchor="mm", stroke_width=pad, stroke_fill=pad_fill
    )

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

def log_gdf_preview(name, gdf, columns, n=5):
    """Log a small preview of selected GeoDataFrame columns for debugging."""
    available_cols = [c for c in columns if c in gdf.columns]
    if not available_cols:
        logger.info("%s columns not found. available=%s", name, list(gdf.columns))
        return

    logger.info("%s columns: %s", name, available_cols)
    if gdf.empty:
        logger.info("%s is empty", name)
        return

    preview = gdf[available_cols].head(n).to_string(index=False)
    logger.info("%s sample rows:\n%s", name, preview)

def split_grids_for_instance(grids, instance_id, num_instances, split_seed):
    """Deterministically shuffle and split grids into disjoint worker chunks."""
    if num_instances <= 0:
        raise ValueError("num_instances must be > 0")
    if not (0 <= instance_id < num_instances):
        raise ValueError(
            f"instance_id must be between 0 and {num_instances - 1}, got {instance_id}"
        )

    shuffled = list(grids)
    random.Random(split_seed).shuffle(shuffled)
    return shuffled[instance_id::num_instances]

def draw_annotations(image, annotations, fontsize=12):
    """Orchestrates drawing, ensuring lines are drawn before polygon labels."""
    image = image.convert("RGBA")
    
    font = ImageFont.truetype(FONT_PATH, fontsize)

    # 1. Draw Rotated Lines First
    #  (Alpha Compositing)
    for ann in [a for a in annotations if a["style"] == "line"]:
        draw_rotated_text_with_padding(
            image, (ann["x"], ann["y"]), ann["text"], 
            ann["angle"], font, "white", "blue", pad=3
        )

    # 2. Draw Static Labels (Polygons) on top
    draw = ImageDraw.Draw(image)
    for ann in [a for a in annotations if a["style"] != "line"]:
        if ann["style"] == "man_made":
            draw_text_with_padding(draw, (ann["x"], ann["y"]), ann["text"], font, "yellow", "black", pad=3)
        else:
            draw_text_with_padding(draw, (ann["x"], ann["y"]), ann["text"], font, "black", "white", pad=3)
    return image

def georef_write(image, center_lon, center_lat, out_path):
    """Write an annotated image as a georeferenced GeoTIFF in EPSG:3857."""
    xmin, ymin, xmax, ymax, res = image_bounds_mercator(
        center_lon, center_lat
    )

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

        # -------- POLYGONS --------
        # Filter first using spatial index (sjoin or intersects) for performance
        possible_polys = poly_gdf[poly_gdf.intersects(bbox_geom)]
        for _, row in possible_polys.iterrows():
            # Filter and collect all valid strings from the specified columns
            valid_tags = [
                str(row[col]) for col in cols 
                if col in row and pd.notna(row[col]) and str(row[col]).lower() != "nan"
            ]
            # Combine them with a comma
            tag = ", ".join(valid_tags) if valid_tags else None
            # Skip if nothing was found
            if not tag: continue

            # Geometrically clip the feature to the visible tile footprint.
            clipped_geom = row.geometry.intersection(bbox_geom)
            if clipped_geom.is_empty or not clipped_geom.is_valid:
                continue
            
            pt = clipped_geom.representative_point()
            px, py = mercator_to_pixel(pt.x, pt.y, cx, cy, IMAGE_SIZE)
            
            if 0 <= px < IMAGE_SIZE[0] and 0 <= py < IMAGE_SIZE[1]:
                annotations.append({
                    "x": px, "y": py, "text": tag, "type": "poly",
                    "style": "man_made" if "man_made" in valid_tags else "poly_other",
                    "angle": None,
                })
        
        # -------- LINES --------
        possible_lines = line_gdf[line_gdf.intersects(bbox_geom)]
        for _, row in possible_lines.iterrows():
            # Filter and collect all valid strings from the specified columns
            valid_tags = [
                str(row[col]) for col in cols 
                if col in row and pd.notna(row[col]) and str(row[col]).lower() != "nan"
            ]
            # Combine them with a comma
            tag = ", ".join(valid_tags) if valid_tags else None
            # Skip if nothing was found
            if not tag: continue

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
                annotations.append({
                    "x": px, "y": py, "text": tag, "type": "line",
                    "style": "man_made" if "man_made" in valid_tags else "line", "angle": angle,
                })

        # -------- DRAW & OUTPUT --------
        image = draw_annotations(image, annotations, fontsize=FONTSIZE)
        if GEOREFERENCED:
            out_path = os.path.join(output_dir, f"bbox_{idx}.tif")
            georef_write(image, cx, cy, out_path)
        else:
            out_path = os.path.join(output_dir, f"bbox_{idx}.png")
            image = image.resize(TARGET_SIZE, resample=Image.LANCZOS)
            ensure_output_dir_for_file(out_path)
            image.save(out_path, dpi=(DPI, DPI))
        return idx, len(annotations), None
    except Exception as e:
        logger.exception("bbox %s failed", idx)
        return idx, 0, str(e)
    
def annotate_bboxes_parallel(bbox_gdf, poly_gdf, cols, line_gdf, line_cols, output_dir, images_dir, files):
    """Dispatch bbox annotation jobs across a thread pool.

    Parameters
    ----------
    bbox_gdf : geopandas.GeoDataFrame
        Grid-cell features to annotate.
    poly_gdf : geopandas.GeoDataFrame
        Polygon annotation features.
    cols : list[str]
        Polygon columns used for text labels.
    line_gdf : geopandas.GeoDataFrame
        Line annotation features.
    line_cols : list[str]
        Line columns used for text labels.
    output_dir : str
        Directory where annotated outputs are written.
    images_dir : str
        Directory containing source imagery.
    files : Any
        Unused compatibility parameter retained by the current call signature.
    """
    logger.info("Queued %s bboxes for annotation", len(bbox_gdf))
    required_cols = {'idx', 'img_idx', 'geometry'}
    missing_cols = required_cols.difference(bbox_gdf.columns)
    if missing_cols:
        raise KeyError(f"bbox_gdf is missing required columns: {sorted(missing_cols)}")

    futures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for _, row in bbox_gdf.iterrows():
            futures.append(
                executor.submit(
                    process_bbox,
                    row['idx'],
                    row.geometry,
                    row['img_idx'],
                    poly_gdf[poly_gdf['grid'] == row['idx']].copy(),
                    cols,
                    line_gdf[line_gdf['grid'] == row['idx']].copy(),
                    line_cols,
                    output_dir,
                    images_dir
                )
            )

        errors = []
        for future in as_completed(futures):
            idx, n, err = future.result()
            if err:
                logger.error("bbox %s failed: %s", idx, err)
                errors.append((idx, err))
            else:
                logger.info("bbox %s done (%s tags)", idx, n)

    if errors:
        first_idx, first_err = errors[0]
        raise RuntimeError(
            f"{len(errors)} of {len(futures)} bbox annotation task(s) failed; "
            f"first failure was bbox {first_idx}: {first_err}"
        )

if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Annotate Bing images for a deterministic subset of grids."
    )
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
    FONTSIZE = int(annotations_cfg["fontsize"])
    DPI = int(annotations_cfg["dpi"])
    target_size_px = int(annotations_cfg["target_size_px"])
    TARGET_SIZE = [target_size_px, target_size_px]
    WEB_MERCATOR_TILE_SIZE = int(annotations_cfg["mercator_tile_size_px"])
    IMAGERY_REFERENCE_TILE_SIZE = int(annotations_cfg["imagery_reference_tile_size_px"])
    EARTH_RADIUS = float(annotations_cfg["earth_radius_m"])
    WORLD_WIDTH = 2 * math.pi * EARTH_RADIUS

    images_dir = cfg["paths"]["annotations_images_dir"]
    grid_filedir = cfg["paths"]["annotations_grid_dir"]
    temp_parquet_dir = cfg["paths"]["annotations_temp_parquet_dir"]
    geojson_file_dir = cfg['paths']['annotations_by_osm_dir']
    file_location_dir = os.path.join(os.path.abspath(os.path.join(grid_filedir, '..')), 'data')
    
    poly_filepath = os.path.join(file_location_dir, 'merged_polygons.parquet')
    line_filepath = os.path.join(file_location_dir, 'merged_lines.parquet')

    points_path = f'{images_dir}/ref.geojson'
    grids_filepath = os.path.join(grid_filedir, f'grids_{os.path.basename(cfg["paths"]["corrected_all_filepath"])}')

    bbox_gdf = gpd.read_file(grids_filepath).to_crs(3857)
    bbox_gdf['idx'] = bbox_gdf['idx'].astype(int)

    points_gdf = gpd.read_file(points_path).to_crs(3857)
    points_gdf['idx'] = points_gdf['idx'].astype(int)
    points_gdf['img_idx'] = points_gdf.index.astype(int)

    bbox_gdf = bbox_gdf.merge(
        points_gdf[['idx', 'img_idx']], 
        on='idx', 
        how='inner'
    )
    bbox_gdf = bbox_gdf[bbox_gdf.geometry.is_valid & ~bbox_gdf.geometry.is_empty]
    logger.info("Prepared %s valid bounding boxes", len(bbox_gdf))
    output_dir = cfg["paths"]["annotated_images_output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    files = set(os.listdir(output_dir))
    geojson_files = set(os.listdir(geojson_file_dir))

    poly_cols = ["id", "man_made", "landuse", "industrial", "power", "resource"]
    line_cols = ["waterway", "man_made", "landuse", "industrial", "power", "resource", "water", "id"]
    cols_string = ','.join(poly_cols)
    line_cols_string = ','.join(line_cols)

    # duckdb_connection owns the scratch database for the whole run; the
    # previous randomly-named temp_*.db was never removed.
    with duckdb_connection() as conn:
        ensure_duckdb_spatial(conn)
        logger.info("DuckDB initialized with SPATIAL extension")

        grids = bbox_gdf['idx'].unique().tolist()
        logger.info("Found %s total grids", len(grids))

        grids = split_grids_for_instance(
            grids,
            instance_id=args.instance_id,
            num_instances=args.num_instances,
            split_seed=args.split_seed,
        )
        logger.info(
            "Instance %s/%s will process %s grids (seed=%s)",
            args.instance_id,
            args.num_instances,
            len(grids),
            args.split_seed,
        )
        if not grids:
            logger.info("No grids assigned to this instance. Exiting.")
            sys.exit(0)

        for i in range(0, len(grids), 2 * MAX_WORKERS):
            sub_grids = grids[i : i + 2 * MAX_WORKERS]
            if not sub_grids:
                continue

            logger.info(
                "Processing grid batch %s-%s (%s grids)",
                i,
                i + len(sub_grids) - 1,
                len(sub_grids),
            )

            # --- Polygons ---
            poly_file_columns = {}
            all_poly_columns = set()
            for grid in sub_grids:
                polygon_file = f"idx_{grid}_polygons.geojson"
                path = f"{geojson_file_dir}/{polygon_file}"
                if polygon_file not in geojson_files:
                    continue
                cols = conn.execute(f"DESCRIBE SELECT * FROM ST_READ('{path}')").df()["column_name"].tolist()
                poly_file_columns[grid] = cols
                all_poly_columns.update(cols)

            all_poly_columns = sorted(all_poly_columns - {"geom"})  # remove geom from attributes

            poly_queries = []
            for grid, cols in poly_file_columns.items():
                polygon_file = f"{geojson_file_dir}/idx_{grid}_polygons.geojson"
                select_cols = []

                for c in all_poly_columns:
                    if c not in poly_cols:
                        continue
                    if c in cols:
                        select_cols.append(c)
                    else:
                        select_cols.append(f"NULL AS {c}")
            
                if select_cols:
                    select_clause = ", ".join(select_cols) + ", "
                    poly_queries.append(f"""
                        SELECT
                            {select_clause}
                            ST_AsText(geom) AS geometry,
                            '{grid}' AS grid
                        FROM ST_READ('{polygon_file}')
                    """)
                else:
                    poly_queries.append(f"""
                        SELECT
                            ST_AsText(geom) AS geometry,
                            '{grid}' AS grid
                        FROM ST_READ('{polygon_file}')
                    """)

            poly_query = " UNION ALL ".join(poly_queries)
            poly_df = conn.execute(poly_query).df()
            poly_df = poly_df.dropna(subset=["geometry"])
            poly_df["geometry"] = poly_df["geometry"].apply(safe_wkt_load)
            poly_gdf = gpd.GeoDataFrame(poly_df, geometry="geometry", crs=4326).to_crs(3857)
            poly_gdf["grid"] = poly_gdf["grid"].astype(int)

            logger.info("Loaded %s polygon records in current batch", len(poly_gdf))
            log_gdf_preview("poly_gdf", poly_gdf, ["grid", "geometry"])

            # --- Lines ---
            line_file_columns = {}
            all_line_columns = set()
            for grid in sub_grids:
                line_file = f"idx_{grid}_lines.geojson"
                path = f"{geojson_file_dir}/{line_file}"
                if line_file not in geojson_files:
                    continue
                cols = conn.execute(f"DESCRIBE SELECT * FROM ST_READ('{path}')").df()["column_name"].tolist()
                line_file_columns[grid] = cols
                all_line_columns.update(cols)

            all_line_columns = sorted(all_line_columns - {"geom"})

            line_queries = []
            for grid, cols in line_file_columns.items():
                line_file = f"{geojson_file_dir}/idx_{grid}_lines.geojson"
                select_cols = []

                for c in all_line_columns:
                    if c not in line_cols:
                        continue
                    if c in cols:
                        select_cols.append(c)
                    else:
                        select_cols.append(f"NULL AS {c}")

                if select_cols:
                    select_clause = ", ".join(select_cols) + ", "
                    line_queries.append(f"""
                        SELECT
                            {select_clause}
                            ST_AsText(geom) AS geometry,
                            '{grid}' AS grid
                        FROM ST_READ('{line_file}')
                    """)
                else:
                    line_queries.append(f"""
                        SELECT
                            ST_AsText(geom) AS geometry,
                            '{grid}' AS grid
                        FROM ST_READ('{line_file}')
                    """)

            line_query = " UNION ALL ".join(line_queries)
            line_df = conn.execute(line_query).df()
            line_df = line_df.dropna(subset=["geometry"])
            line_df["geometry"] = line_df["geometry"].apply(safe_wkt_load)
            line_gdf = gpd.GeoDataFrame(line_df, geometry="geometry", crs=4326).to_crs(3857)
            line_gdf["grid"] = line_gdf["grid"].astype(int)

            logger.info("Loaded %s line records in current batch", len(line_gdf))
            log_gdf_preview("lines_gdf", line_gdf, ["grid", "geometry"])
            batch_bbox_gdf = bbox_gdf[bbox_gdf['idx'].isin(sub_grids)].copy()

            annotate_bboxes_parallel(batch_bbox_gdf, poly_gdf, ["man_made", "landuse", "industrial", "power", "resource", "water"], line_gdf, ["waterway", "man_made", "landuse", "industrial", "power", "resource", "water"], 
                            output_dir, images_dir, files)
        logger.info("Annotation pipeline finished")