"""Render a vector layer into one georeferenced image per annotation grid cell.

Reads the grid written by NEW_01_GENERATEGRIDS (``grids_<corrected_all name>``
in ``annotations_grid_dir``) and the layer at ``export_layer_filepath``, and
writes ``<id_field>.<image_format>`` plus a ``.wld`` world file per cell into
``export_patch_dir``.
"""

import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..utils import configure_logging, ensure_output_dir_for_file

IMAGE_FORMATS = ("tif", "png")


def world_file_lines(extent, width_px, height_px):
    """The six lines of an ESRI world file for an image covering ``extent`` (minx, miny, maxx, maxy)."""
    return [(extent[2] - extent[0]) / width_px, 0.0, 0.0, -(extent[3] - extent[1]) / height_px, extent[0], extent[3]]


def export_patches(grid, layers, output_dir, id_field, width_px, height_px, dpi, image_format):
    """Write one image and world file per grid cell; return the image paths."""
    if id_field not in grid.columns:
        raise KeyError(f"The field '{id_field}' does not exist in the grid layer.")
    if image_format not in IMAGE_FORMATS:
        raise ValueError(f"image_format must be one of {IMAGE_FORMATS}, got {image_format!r}")
    written = []
    for _, feature in grid.iterrows():
        extent = feature.geometry.bounds
        fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        for layer in layers:
            layer.plot(ax=ax, color="lightblue", edgecolor="k")
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
        ax.axis("off")
        file_path = os.path.join(output_dir, f"{feature[id_field]}.{image_format}")
        ensure_output_dir_for_file(file_path)
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        with open(os.path.splitext(file_path)[0] + ".wld", "w") as wf:
            wf.write("".join(f"{v}\n" for v in world_file_lines(extent, width_px, height_px)))
        written.append(file_path)
    return written


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run NEW_04_EXPORTGEOTIFF.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="NEW_04_EXPORTGEOTIFF", **overrides)
    paths, export = cfg["paths"], cfg["export"]
    grid_path = os.path.join(paths["annotations_grid_dir"], f"grids_{os.path.basename(paths['corrected_all_filepath'])}")
    grid = gpd.read_file(grid_path)
    layer = gpd.read_file(paths["export_layer_filepath"]).to_crs(grid.crs)
    written = export_patches(grid, [layer], paths["export_patch_dir"], export["id_field"],
                             int(export["output_width_px"]), int(export["output_height_px"]), int(export["dpi"]),
                             export["image_format"])
    print(f"Exported {len(written)} georeferenced patches to {paths['export_patch_dir']}")


if __name__ == "__main__":
    configure_logging()
    main()
