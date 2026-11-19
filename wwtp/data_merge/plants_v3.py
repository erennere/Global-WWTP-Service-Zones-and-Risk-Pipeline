"""Plant list v3: v2 updated with the work list and the results of its new images.

Settings: config.yaml, section ``plants_v3``. Command: ``python -m wwtp.data_merge.plants_v3``.

Steps, from ``base_filepath`` (all_annotated_v2):
1. Plants without a valid location are left out.
2. The plants the work list removed (``removed_plants``: not in the corrected EU file) are left out.
3. Plants of the work list that the corrected EU file moved go to the file's location (the v2 location
   stays in ``geometry_v2``). A move longer than ``wrong_code_shift_m`` means the plant carried another
   plant's EU code (a v2 merge error, e.g. a Hungarian plant with a French code): the v2 plant stays
   where it is, without the EU file's columns, and the file plant becomes a plant of its own.
4. The new plants of the work list (national, corrected EU file) are added.
5. The results of the new images replace the tank detections (``detection_columns``, from the
   segmentation CSV) of the work-list plants that have them, and the site category (``category_columns``,
   from the image-to-text CSV, when ``with_image_text``) of the new plants (``image_text_list``). Segmentation
   names an image ``<n>.png``: the ref.geojson feature whose idx is n, that idx the plant's row in
   ``images_ref_list`` (the list the images were made from). A result is used only when the work list has
   that plant (same idx and plant_id) within ``max_image_offset_m`` of the image; the others are written to
   ``unused_image_results`` with the reason. Image-to-text names an image ``bbox_<idx>.png``. A plant without
   a result keeps its v2 values (``size_source`` v2_image, or none); old plants keep their v2 category.
6. Plants closer than ``same_site_distance_m`` are one plant: the location of the member whose
   ``location_source`` comes first in ``location_priority``, and per column the first value in that order;
   the largest POP_SERVED; ``idx_v2`` and ``origin`` of the member with the lowest idx_v2 (a v2 plant).
7. Every plant gets a new ``idx`` (0..n-1); ``idx_v2`` keeps its v2 number (for new plants their
   work-list number, which follows v2's).

Added columns: ``origin`` (v2, new_plant_national, new_plant_eu_file), ``location_source`` (v2,
corrected_eu_file, national_plants), ``size_source`` (v2_image, new_image, none, no_image: a work-list plant
the images were not made for), ``new_image`` (the
image of the new results), ``location_shift_m``, ``geometry_v2`` (WKT), ``idx_v2``, ``same_site_plants``
(how many plants step 6 merged into this one).
Writes ``paths.output``, ``paths.summary`` (counts per step) and ``paths.unused_image_results``.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import Geod
from scipy.spatial import cKDTree

from ..annotation_scripts.merge_annotations import decode_gen_text, parse_idx_from_image_name
from ..geo_utils import UnionFind, parse_diameters_to_round_area
from ..service_areas.size import NO_IMAGE
from ..starter import load_config
from ..utils import atomic_output, configure_logging
from .work_list import file_countries

logger = logging.getLogger(__name__)
GEOD = Geod(ellps="WGS84")
# Candidate pairs come from chords on a sphere of this radius, padded by 1 %; every pair is then checked
# on the WGS84 ellipsoid, which differs from the sphere by less than 0.6 %.
EARTH_RADIUS_M = 6_371_008.8
ADDED = ["origin", "location_source", "size_source", "new_image", "location_shift_m", "geometry_v2", "idx_v2",
         "same_site_plants"]  # fmt: skip


def valid(gdf):
    return gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid


def metres(a, b):
    """Geodesic metres between two point series, row by row."""
    return GEOD.inv(a.x.to_numpy(), a.y.to_numpy(), b.x.to_numpy(), b.y.to_numpy())[2]


def image_results(cfg, work):
    """Detections and categories of the new images, by work-list idx, and the segmentation rows not used.

    A segmentation row is used only when its image shows the plant it is given to: ``<n>.png`` is the
    ref.geojson feature whose idx is n (``image_names`` ref_idx; ref_row: the n-th feature), that idx is a row
    of ``images_ref_list`` (the list the images were made from: plant idx and plant_id), and the work list has
    that idx with the same plant_id, within ``max_image_offset_m`` of the image's point. Any other row is set
    apart with its reason, never given to a plant. Site categories (image-to-text, ``bbox_<idx>.png``) are taken
    only for the plants of ``image_text_list`` (the new plants; the others keep their v2 category).
    """
    paths = cfg["paths"]
    listing = pd.read_csv(paths["images_ref_list"], dtype={"plant_id": str})
    ref = gpd.read_file(paths["images_ref"]).to_crs(4326)
    ref_list_row = ref["idx"].astype(int).to_numpy()

    seg = pd.read_csv(paths["segmentation_results"])
    n = pd.to_numeric(seg["img_name"].astype(str).str.split(".").str[0], errors="coerce")
    if cfg["image_names"] == "ref_idx":
        feature = pd.Series(np.arange(len(ref)), index=ref_list_row).reindex(n.to_numpy()).to_numpy()
    else:
        feature = n.where(n.between(0, len(ref) - 1)).to_numpy()
    reason = pd.Series(None, index=seg.index, dtype=object)
    reason[np.isnan(feature)] = "no ref.geojson feature for this image"
    has = ~np.isnan(feature)
    list_row = np.full(len(seg), -1)
    list_row[has] = ref_list_row[feature[has].astype(int)]
    in_list = has & (list_row >= 0) & (list_row < len(listing))
    reason[has & ~in_list] = "ref.geojson idx outside images_ref_list"
    seg["idx"] = pd.array([pd.NA] * len(seg), dtype="Int64")
    seg.loc[in_list, "idx"] = listing["idx"].to_numpy()[list_row[in_list]]
    seg["plant_id"] = None
    seg.loc[in_list, "plant_id"] = listing["plant_id"].to_numpy()[list_row[in_list]]

    plants = work.drop_duplicates("idx").set_index("idx")
    known = in_list & seg["idx"].isin(plants.index).to_numpy()
    reason[in_list & ~known] = "plant not in the work list"
    same_id = known.copy()
    same_id[known] = (plants["plant_id"].astype(str).reindex(seg.loc[known, "idx"]).to_numpy()
                      == seg.loc[known, "plant_id"].astype(str).to_numpy())  # fmt: skip
    reason[known & ~same_id] = "the work list has another plant at this idx"
    near = same_id.copy()
    if same_id.any():
        at = plants.geometry.reindex(seg.loc[same_id, "idx"]).to_numpy()
        pt = ref.geometry.to_numpy()[feature[same_id].astype(int)]
        xy = np.array([[a.x, a.y, b.x, b.y] for a, b in zip(at, pt)], dtype=float).reshape(-1, 4)
        dist = GEOD.inv(xy[:, 0], xy[:, 1], xy[:, 2], xy[:, 3])[2]
        near[same_id] = dist <= float(cfg["max_image_offset_m"])
    reason[same_id & ~near] = f"image farther than {cfg['max_image_offset_m']:g} m from the plant"
    seg["reason"] = reason
    unused = seg[~near].copy()
    seg = seg[near].copy()
    seg["idx"] = seg["idx"].astype(int)
    seg["new_image"] = seg["img_name"].astype(str)
    seg = seg[["idx", "new_image", *cfg["detection_columns"]]].drop_duplicates("idx")

    cats = list(cfg["category_columns"])
    if cfg["with_image_text"]:
        text = pd.read_csv(paths["image_text_results"])
        text["idx"] = text["image"].apply(parse_idx_from_image_name)
        new_plants = pd.read_csv(paths["image_text_list"], usecols=["idx"])["idx"].astype(int)
        text = text[text["idx"].notna() & text["idx"].isin(new_plants)].astype({"idx": int})
        text[cats] = pd.DataFrame([decode_gen_text(t) for t in text["gen_text"]], index=text.index, columns=cats)
        text = text[["idx", *cats]].drop_duplicates("idx")
    else:  # the image-to-text results do not exist yet
        text = pd.DataFrame({"idx": pd.Series(dtype=int), **{c: pd.Series(dtype=object) for c in cats}})
    results = seg.merge(text, on="idx", how="outer")
    logger.info("new images: %d segmentation results used, %d set apart %s; %d with a site category", len(seg),
                len(unused), unused["reason"].value_counts().to_dict(), len(text))  # fmt: skip
    return results, unused


def update_sizes(df, results, cfg):
    """Detections and categories of the new images for the rows (by ``wl_idx``) that have them."""
    res = results.set_index("idx")
    cats = list(cfg["category_columns"])
    for cols, done in (
        ([*cfg["detection_columns"], "new_image"], res.index[res["new_image"].notna()]),
        (cats, res.index[res[cats].notna().any(axis=1)]),
    ):
        rows = df["wl_idx"].isin(done)
        for c in cols:
            df.loc[rows, c] = res[c].reindex(df.loc[rows, "wl_idx"]).to_numpy()
    seg_done = df["wl_idx"].isin(res.index[res["new_image"].notna()])
    df.loc[seg_done, "size_source"] = "new_image"
    if "round_area" in df.columns:
        df.loc[seg_done, "round_area"] = df.loc[seg_done, "diameters"].apply(parse_diameters_to_round_area)
    return df


def same_sites(gdf, limit):
    """Groups of rows closer than ``limit`` metres (geodesic), as lists of positions."""
    lon, lat = np.radians(gdf.geometry.x.to_numpy()), np.radians(gdf.geometry.y.to_numpy())
    xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    chord = 2 * np.sin(limit * 1.01 / EARTH_RADIUS_M / 2)
    pairs = np.array(sorted(cKDTree(xyz).query_pairs(chord)), dtype=int).reshape(-1, 2)
    if len(pairs):
        p = gdf.geometry.iloc
        d = GEOD.inv(p[pairs[:, 0]].x.to_numpy(), p[pairs[:, 0]].y.to_numpy(),
                     p[pairs[:, 1]].x.to_numpy(), p[pairs[:, 1]].y.to_numpy())[2]  # fmt: skip
        pairs = pairs[d <= limit]
    uf = UnionFind(len(gdf))
    for i, j in pairs:
        uf.union(int(i), int(j))
    groups = {}
    for i in range(len(gdf)):
        groups.setdefault(uf.find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def merge_same_sites(gdf, cfg):
    """One row per same-site group: location and first values by location_priority."""
    rank = {s: i for i, s in enumerate(cfg["location_priority"])}
    groups = same_sites(gdf, float(cfg["same_site_distance_m"]))
    gdf["same_site_plants"] = 1
    drop, rows = [], []
    for g in groups:
        sub = gdf.iloc[g]
        sub = sub.iloc[np.argsort([rank[s] for s in sub["location_source"]], kind="stable")]
        merged = sub.bfill().iloc[0].copy()
        merged["geometry"] = sub.geometry.iloc[0]
        merged["same_site_plants"] = len(sub)
        oldest = sub.iloc[np.argmin(sub["idx_v2"].fillna(np.inf).to_numpy())]  # a v2 plant before a new one
        merged["idx_v2"], merged["origin"] = oldest["idx_v2"], oldest["origin"]
        if "POP_SERVED" in sub:
            merged["POP_SERVED"] = pd.to_numeric(sub["POP_SERVED"], errors="coerce").max()
        rows.append(merged)
        drop.extend(sub.index)
    kept = gdf.drop(index=drop)
    out = pd.concat([kept, pd.DataFrame(rows)]) if rows else kept
    logger.info("same sites: %d groups of %d plants merged", len(groups), len(drop))
    return gpd.GeoDataFrame(out, geometry="geometry", crs=4326), len(groups), len(drop)


def build(cfg):
    paths = cfg["paths"]
    counts = {}
    base = gpd.read_file(paths["base_filepath"]).to_crs(4326)
    counts["v2 plants"] = len(base)
    base = base[valid(base)].copy()
    counts["v2 plants with a location"] = len(base)
    columns = list(base.columns.drop("geometry"))
    base["idx"] = base["idx"].astype(int)
    if base["idx"].duplicated().any():
        raise ValueError(f"{paths['base_filepath']}: idx must be unique")
    base["wl_idx"], base["idx_v2"] = pd.NA, base["idx"]
    base["origin"], base["location_source"], base["size_source"] = "v2", "v2", "v2_image"

    removed = pd.read_csv(paths["removed_plants"])["idx"].astype(int)
    if not removed.isin(base["idx"]).all():
        raise ValueError(f"{paths['removed_plants']}: plants not in {paths['base_filepath']}")
    base = base[~base["idx"].isin(removed)]
    counts["removed (not in the corrected EU file)"] = len(removed)

    work = gpd.read_file(paths["work_list"]).to_crs(4326)
    work["idx"] = work["idx"].astype(int)
    old = work[work["origin"] == "without_heigit_size"]
    missing = ~old["idx"].isin(base["idx"])
    if missing.any():
        raise ValueError(f"{paths['work_list']}: {int(missing.sum())} plants without size are not in the v2 list")
    base = base.set_index("idx", drop=False)
    base.loc[old["idx"], "wl_idx"] = old["idx"].to_numpy()

    moved = old[old["location_source"] == "corrected_eu_file"].set_index("idx")
    wrong = moved.index[moved["location_shift_m"] > float(cfg["wrong_code_shift_m"])]
    moved = moved.drop(index=wrong)
    base["geometry_v2"] = None
    base.loc[moved.index, "geometry_v2"] = base.loc[moved.index].geometry.to_wkt(rounding_precision=-1)
    base.loc[moved.index, "geometry"] = moved.geometry
    base.loc[moved.index, "location_shift_m"] = moved["location_shift_m"]
    base.loc[moved.index, "location_source"] = "corrected_eu_file"
    counts["moved to the corrected EU file's location"] = len(moved)

    # a plant with another plant's EU code: it stays, the file plant becomes a plant of its own
    eu_columns = [c for c in pyogrio.read_info(paths["eu_file"])["fields"] if c in columns and c not in ("lat", "lon")]
    split = work[work["idx"].isin(wrong)].copy()
    split = split[[c for c in columns if c in eu_columns or c == "ISO_2"] + ["idx", "geometry"]]
    split["ISO_2"] = file_countries(split, cfg)
    split["origin"], split["location_source"] = "new_plant_eu_file", "corrected_eu_file"
    base.loc[wrong, eu_columns] = None
    base.loc[wrong, "wl_idx"] = pd.NA
    counts["plants with another plant's EU code (file plant added)"] = len(wrong)

    new = work[work["origin"] != "without_heigit_size"].copy()
    national = new["origin"] == "new_plant_national"
    new["WASTE_ID"] = new["WASTE_ID"].astype(object).where(~national, new["OLD_WASTE_ID"])  # back from the work list
    new["idx_v2"] = new["idx"]
    split["idx_v2"] = pd.NA  # its work-list number is the v2 plant's, which stays too
    new = pd.concat([new, split], ignore_index=True)
    new["wl_idx"], new["size_source"] = new["idx"], "none"
    new = new[[c for c in columns if c in new.columns] + ["wl_idx", "idx_v2", "origin", "location_source",
                                                         "size_source", "geometry"]]  # fmt: skip
    counts.update({f"added: {k}": v for k, v in new["origin"].value_counts().items()})

    plants = pd.concat([base.reset_index(drop=True), new], ignore_index=True)
    plants = gpd.GeoDataFrame(plants, geometry="geometry", crs=4326)
    results, unused = image_results(cfg, work)
    plants = update_sizes(plants, results, cfg)
    counts["plants with a new image (segmentation results)"] = int((plants["size_source"] == "new_image").sum())
    # work-list plants the images were not made for (added after them): size_source no_image, so that the
    # plant size takes their country's median, not the low value of plants seen without tanks
    imaged = pd.read_csv(paths["images_ref_list"], usecols=["idx"])["idx"].astype(int)
    never = (plants["size_source"] == "none") & plants["wl_idx"].notna() & ~plants["wl_idx"].isin(imaged)
    plants.loc[never, "size_source"] = NO_IMAGE
    counts["plants never imaged (size_source no_image)"] = int(never.sum())
    counts["segmentation results set apart (image not verified to show the plant)"] = len(unused)

    plants, n_groups, n_rows = merge_same_sites(plants.reset_index(drop=True), cfg)
    counts[f"same-site groups merged (closer than {cfg['same_site_distance_m']:g} m)"] = n_groups
    counts["plants in them"] = n_rows

    plants = plants.sort_values("idx_v2", kind="stable").reset_index(drop=True)
    plants["idx"], plants["idx_v2"] = np.arange(len(plants)), plants["idx_v2"].astype("Int64")
    plants = plants[[*columns, *[c for c in ADDED if c not in columns], "geometry"]]
    counts["v3 plants"] = len(plants)
    return gpd.GeoDataFrame(plants, geometry="geometry", crs=4326), counts, unused


def write(cfg, plants, counts, unused):
    with atomic_output(cfg["paths"]["output"]) as tmp:
        plants.to_file(tmp, driver="GPKG", engine="pyogrio")
    table = pd.DataFrame({"step": list(counts), "plants": list(counts.values())})
    with atomic_output(cfg["paths"]["summary"]) as tmp:
        table.to_csv(tmp, index=False)
    with atomic_output(cfg["paths"]["unused_image_results"]) as tmp:
        unused.to_csv(tmp, index=False)


def main():
    cfg = load_config(script_name="plants_v3")
    plants, counts, unused = build(cfg)
    write(cfg, plants, counts, unused)
    for step, n in counts.items():
        logger.info("%s: %s", step, n)


if __name__ == "__main__":
    configure_logging()
    main()
