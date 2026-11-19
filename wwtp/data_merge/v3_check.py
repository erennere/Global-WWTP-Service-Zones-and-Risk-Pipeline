"""Check, before the run, everything the text overlay of the new plants and plants_v3 will read.

Command: ``python -m wwtp.data_merge.v3_check`` (settings: config.yaml, sections download_bing_annotate,
NEW_01_GENERATEGRIDS and plants_v3, with the cluster overlay). Reads, never writes (apart from a test file
that is removed again), and prints one line per check: ``OK``, ``NOTE`` (works, worth knowing) or
``PROBLEM`` (the run would go wrong), then ``READY`` or ``NOT READY``.

Overlay: the images' ref.geojson against the list they were made from (plant_id and place per row), every
``<n>.png`` there and its size, every image's plant in the work list, the OSM tile found for every image
(by place), the OSM features of a sample of images, the output folders writable.
plants_v3: its input files, every segmentation result linked to the plant its image shows, and the whole
build in memory (counts of every step, plants never imaged).
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image

from ..annotation_scripts import download_bing_annotate as dba
from ..annotation_scripts.images_ref import image_offsets_m, plant_idx_of_rows
from ..service_areas.size import NO_IMAGE
from ..starter import load_config
from . import plants_v3 as p3

SAMPLE_IMAGES = 200  # images whose OSM features are read; the run reads all of them
_problems = []


def say(level, text):
    if level == "PROBLEM":
        _problems.append(text)
    print(f"{level:8s}{text}", flush=True)


def check(level_if_false, ok, text):
    say("OK" if ok else level_if_false, text)
    return ok


def writable(folder):
    """True when a file can be written in ``folder`` (created if missing, the test file removed again)."""
    try:
        os.makedirs(folder, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".v3_check_", delete=True):
            pass
        return True
    except OSError:
        return False


def overlay():
    cfg = load_config(script_name="download_bing_annotate")
    ann, paths = cfg["annotations"], cfg["paths"]
    tol = float(cfg["max_image_offset_m"])
    # the overlay's settings, as its __main__ block sets them
    dba.IMAGE_SIZE = [int(ann["image_size_px"])] * 2
    dba.IMAGERY_REFERENCE_TILE_SIZE = int(ann["imagery_reference_tile_size_px"])
    dba.WEB_MERCATOR_TILE_SIZE = int(ann["mercator_tile_size_px"])
    dba.BASE_Z17_RES = float(ann["base_z17_resolution"])
    dba.EARTH_RADIUS = float(ann["earth_radius_m"])
    images_dir = paths["annotations_images_dir"]
    print(f"\n== Overlay of the new plants: {images_dir}")
    print(f"        {ann['image_size_px']} px images at {dba.pixel_size_m():.3f} m/px = "
          f"{dba.IMAGE_SIZE[0] * dba.pixel_size_m():.0f} m; "
          f"<n>.png = the ref.geojson feature {'whose idx is n' if ann['image_names'] == 'ref_idx' else 'number n'}")  # fmt: skip

    ref_path = os.path.join(images_dir, "ref.geojson")
    if not check("PROBLEM", os.path.exists(ref_path), f"ref.geojson there ({ref_path})"):
        return
    ref = gpd.read_file(ref_path).to_crs(3857)
    rows = ref["idx"].astype(int)
    check("PROBLEM", rows.is_unique and rows.min() == 0 and rows.max() == len(ref) - 1,
          f"ref.geojson: {len(ref)} features, idx 0..{len(ref) - 1} each once")  # fmt: skip
    listing = pd.read_csv(paths["images_ref_list"], dtype={"plant_id": str})
    if not check("PROBLEM", len(listing) == len(ref),
                 f"images_ref_list {paths['images_ref_list']}: {len(listing)} rows for {len(ref)} images"):  # fmt: skip
        return
    by_row = listing.iloc[rows.to_numpy()].reset_index(drop=True)
    if "plant_id" in ref.columns:
        same = (ref["plant_id"].astype(str).to_numpy() == by_row["plant_id"].astype(str).to_numpy()).sum()
        check("PROBLEM", same == len(ref), f"every image's plant_id = the list's at its row: {same} of {len(ref)}")
    pts = ref.geometry.to_crs(4326)
    d = dba.GEOD.inv(pts.x.to_numpy(), pts.y.to_numpy(), by_row["lon"].to_numpy(), by_row["lat"].to_numpy())[2]
    check("PROBLEM", (d <= tol).all(), f"every image at its list position (up to {d.max():.2f} m, limit {tol:g} m)")

    images = plant_idx_of_rows(ref.assign(ref_row=np.arange(len(ref))), paths["images_ref_list"])
    images["img_idx"] = images["list_row" if ann["image_names"] == "ref_idx" else "ref_row"]
    files = set(os.listdir(images_dir))
    missing = [n for n in images["img_idx"] if f"{int(n)}.png" not in files]
    check("PROBLEM", not missing, f"image files: {len(images) - len(missing)} of {len(images)} there"
          + (f" (missing e.g. {missing[:5]})" if missing else ""))  # fmt: skip
    sizes = {}
    for n in images["img_idx"]:
        path = os.path.join(images_dir, f"{int(n)}.png")
        if os.path.exists(path):
            with Image.open(path) as im:
                sizes[im.size] = sizes.get(im.size, 0) + 1
    want = (int(ann["image_size_px"]),) * 2
    check("PROBLEM", set(sizes) <= {want}, f"image sizes: {sizes} (expected {want[0]} px)")

    plants = gpd.read_file(paths["annotation_points_filepath"], columns=["idx", "plant_id"])
    plants["idx"] = plants["idx"].astype(int)
    off = image_offsets_m(images, plants)
    check("PROBLEM", np.isfinite(off).all() and (off <= tol).all(),
          f"every image's plant in the work list {paths['annotation_points_filepath']} "
          f"({int(np.isfinite(off).sum())} of {len(images)}; up to {np.nanmax(off):.2f} m)")  # fmt: skip

    grid = os.path.join(paths["annotations_grid_dir"], f"grids_{os.path.basename(paths['annotation_points_filepath'])}")
    osm_dir = paths["annotations_by_osm_dir"]
    if not check("PROBLEM", os.path.exists(grid) and os.path.isdir(osm_dir), f"tiles {grid} and OSM files {osm_dir}"):
        return
    tiles = gpd.read_file(grid).to_crs(3857)
    tiles = tiles[tiles.geometry.notna() & ~tiles.geometry.is_empty].copy()
    tiles["idx"] = tiles["idx"].astype(int)
    osm_files = set(os.listdir(osm_dir))
    match = dba.match_osm_tiles(images, tiles, osm_files, tol)
    sources = match["osm_source"].value_counts().to_dict()
    say("OK" if sources.get("own_tile", 0) == len(images) else "NOTE",
        f"OSM tile of every image, found by place: {sources} (own_tile: the plant's own tile; tile_at_image: "
        "tiles numbered otherwise, still the right place; overlapping_tiles/none: no tile made for it)")  # fmt: skip
    check("PROBLEM", sources.get("none", 0) == 0, f"images without any OSM tile: {sources.get('none', 0)}")

    images = images.join(match)
    dba._TILES.update(osm_dir=osm_dir, osm_files=osm_files,
                      poly_cols=["id", "man_made", "landuse", "industrial", "power", "resource"],
                      line_cols=["waterway", "man_made", "landuse", "industrial", "power", "resource", "water", "id"])  # fmt: skip
    sample = images[images["osm_tiles"].map(len) > 0].sample(min(SAMPLE_IMAGES, len(images)), random_state=1)
    read_errors, with_features, on_image = [], 0, 0
    for _, im in sample.iterrows():
        frame = dba.image_frame(im.geometry.x, im.geometry.y)
        try:
            polys = dba.read_osm_of_image(im["osm_tiles"], "polygons", dba._TILES["poly_cols"])
            lines = dba.read_osm_of_image(im["osm_tiles"], "lines", dba._TILES["line_cols"])
        except Exception as e:
            read_errors.append((int(im["idx"]), str(e)[:80]))
            continue
        n = len(polys) + len(lines)
        with_features += n > 0
        on_image += n > 0 and bool(polys.intersects(frame).any() or lines.intersects(frame).any())
    check("PROBLEM", not read_errors, f"OSM files of {len(sample)} sample images readable"
          + (f" (errors e.g. {read_errors[:3]})" if read_errors else ""))  # fmt: skip
    check(
        "PROBLEM",
        on_image == with_features,
        f"sample: {with_features} images with OSM features, {on_image} of them with features on the image",
    )
    check(
        "PROBLEM",
        writable(paths["annotated_images_output_dir"]),
        f"output folder writable: {paths['annotated_images_output_dir']}",
    )
    check("PROBLEM", writable(paths["match_report_dir"]), f"match report folder writable: {paths['match_report_dir']}")


def v3():
    cfg = load_config(script_name="plants_v3")
    paths = cfg["paths"]
    print(f"\n== plants_v3 -> {paths['output']}")
    inputs = ["base_filepath", "work_list", "removed_plants", "eu_file", "images_ref", "images_ref_list",
              "segmentation_results"] + (["image_text_results", "image_text_list"] if cfg["with_image_text"] else [])  # fmt: skip
    there = [k for k in inputs if os.path.exists(paths[k])]
    for k in inputs:
        check("PROBLEM", k in there, f"{k}: {paths[k]}")
    if len(there) < len(inputs):
        return
    check(
        "PROBLEM",
        writable(os.path.dirname(paths["output"])),
        f"output folder writable: {os.path.dirname(paths['output'])}",
    )
    work = gpd.read_file(paths["work_list"]).to_crs(4326)
    work["idx"] = work["idx"].astype(int)
    results, unused = p3.image_results(cfg, work)
    seg_rows = len(pd.read_csv(paths["segmentation_results"], usecols=["img_name"]))
    check("PROBLEM", len(unused) == 0,
          f"segmentation results linked to the plant their image shows: {len(results)} of {seg_rows}"
          + (f"; set apart {unused['reason'].value_counts().to_dict()}" if len(unused) else ""))  # fmt: skip
    tanks = int(((pd.to_numeric(results["num_detection_circle"], errors="coerce") > 0)
                 | (pd.to_numeric(results["num_detection_rect"], errors="coerce") > 0)).sum())  # fmt: skip
    say("OK", f"of them with tanks: {tanks}")

    plants, counts, unused = p3.build(cfg)
    print("        the whole build, in memory (nothing written):")
    for step, n in counts.items():
        print(f"          {step}: {n}")
    check(
        "PROBLEM",
        plants["idx"].is_unique and plants.geometry.notna().all(),
        f"v3: {len(plants)} plants, idx unique, all located",
    )
    never = plants[plants["size_source"] == NO_IMAGE]
    medians = load_config(script_name="create_voronoi")["footprint_to_people"]["no_image_people_by_country"]
    without = sorted(set(never["ISO_2"].dropna()) - set(medians))
    say("OK" if not without else "NOTE",
        f"plants never imaged: {len(never)} {never['ISO_2'].value_counts().to_dict()}; their country's median "
        f"{ {c: medians[c] for c in sorted(set(never['ISO_2'].dropna()) & set(medians))} }"
        + (f"; no median for {without} (they take the low value)" if without else ""))  # fmt: skip


def main():
    import geopandas
    import pyogrio
    import shapely

    print(f"python {sys.version.split()[0]}, geopandas {geopandas.__version__}, shapely {shapely.__version__}, "
          f"pyogrio {pyogrio.__version__}")  # fmt: skip
    check("PROBLEM", int(shapely.__version__.split(".")[0]) >= 2, "shapely 2 or newer")
    for part in (overlay, v3):
        try:
            part()
        except Exception:
            say("PROBLEM", f"{part.__name__} stopped:\n{traceback.format_exc()}")
    print("\n" + ("READY" if not _problems else f"NOT READY: {len(_problems)} problem(s)"))
    return 0 if not _problems else 1


if __name__ == "__main__":
    sys.exit(main())
