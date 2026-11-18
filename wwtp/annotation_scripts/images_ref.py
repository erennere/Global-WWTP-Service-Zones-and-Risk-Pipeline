"""Checks that a set of images belongs to the annotated plants, before anything is drawn or merged.

The images of a set are ``<row>.png`` with ``ref.geojson`` (row = image). Its column ``idx`` is the row
(from 0) of the plant in the list the images were made from (``images_ref_list``: Sukanya numbered the
plants of the work list she got, 17 Nov 2026); ``plant_idx_of_rows`` turns it into the plant's ``idx``.
Both the text overlay and plants_v3 join by that ``idx``; a ``ref.geojson`` made from another plant list
would put every label and size on the wrong plant.
"""

from __future__ import annotations

import os

import geopandas as gpd
import pandas as pd
from PIL import Image
from pyproj import Geod

GEOD = Geod(ellps="WGS84")


def plant_idx_of_rows(ref, list_path):
    """``ref`` with ``idx`` = the plant's idx; ``ref.geojson``'s own ``idx`` (the row, from 0, of the plant in
    the list the images were made from) is kept as ``list_row``."""
    numbers = pd.read_csv(list_path, usecols=["idx"])["idx"].astype(int).to_numpy()
    rows = ref["idx"].astype(int)
    outside = ((rows < 0) | (rows >= len(numbers))).to_numpy()
    if outside.any():
        raise ValueError(f"{int(outside.sum())} images of ref.geojson have a row outside the {len(numbers)} rows of "
                         f"{list_path} (e.g. {rows[outside].head(5).tolist()})")  # fmt: skip
    out = ref.copy()
    out["list_row"] = rows
    out["idx"] = numbers[rows.to_numpy()]
    return out


def check_images_ref(ref, points, max_offset_m, name):
    """Raise unless every image's point lies within ``max_offset_m`` of its plant (same ``idx``)."""
    ref = ref.to_crs(4326)
    points = points.to_crs(4326).drop_duplicates("idx").set_index("idx")
    unknown = ~ref["idx"].isin(points.index)
    if unknown.any():
        raise ValueError(f"{name}: {int(unknown.sum())} images of plants that are not in the plant list "
                         f"(e.g. idx {ref.loc[unknown, 'idx'].head(5).tolist()})")  # fmt: skip
    at = gpd.GeoSeries(points.geometry.reindex(ref["idx"]).to_numpy(), crs=4326).centroid
    pt = ref.geometry.centroid
    offset = GEOD.inv(at.x.to_numpy(), at.y.to_numpy(), pt.x.to_numpy(), pt.y.to_numpy())[2]
    off = offset > float(max_offset_m)
    if off.any():
        raise ValueError(f"{name}: {int(off.sum())} of {len(ref)} images are farther than {max_offset_m:g} m from "
                         f"their plant (up to {offset.max():.0f} m): the plant list changed after the images "
                         "were taken")  # fmt: skip


def check_image_size(images_dir, rows, size_px):
    """Raise unless the first image of ``rows`` there is ``size_px`` x ``size_px`` pixels."""
    for row in rows:
        path = os.path.join(images_dir, f"{row}.png")
        if os.path.exists(path):
            with Image.open(path) as image:
                if image.size != (size_px, size_px):
                    raise ValueError(f"{path} is {image.size[0]} x {image.size[1]} pixels; the overlay expects "
                                     f"{size_px} x {size_px} (download_bing_annotate.annotations.image_size_px)")  # fmt: skip
            return
    raise FileNotFoundError(f"none of the images listed in {images_dir}/ref.geojson is there")
