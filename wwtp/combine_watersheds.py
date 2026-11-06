"""Merge watershed archive contents into a single GeoPackage layer.

The script scans a directory of zip archives, extracts the first readable
geospatial layer from each archive, concatenates the results, and writes a
combined watershed dataset for the configured level.
"""

import argparse
import os
import zipfile
import tempfile
from pathlib import Path
import geopandas as gpd
import pandas as pd
from .starter import add_standard_override_arguments, load_config, parse_config_overrides
from .utils import atomic_output, configure_logging

def extract_and_merge_geodata(zip_dir, output_path, output_filename="merged.gpkg"):
    """Extract readable geospatial files from zip archives and merge them."""
    zip_dir = Path(zip_dir)
    output_path = Path(output_path)
    if not zip_dir.exists() or not zip_dir.is_dir():
        raise FileNotFoundError(f"Zip directory not found: {zip_dir}")
    output_path.mkdir(parents=True, exist_ok=True)

    merged_frames = []
    zips_without_layer = []

    for zip_file in zip_dir.glob("*.zip"):
        print(f"Processing {zip_file.name}")
        added_from_zip = False
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # A corrupt archive is a hard error: silently skipping it produced a
            # merged layer that was quietly missing a whole watershed level.
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(tmpdir)

            # Walk through extracted contents to find geographic files
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    filepath = Path(root) / file
                    try:
                        # Try reading with geopandas
                        gdf = gpd.read_file(filepath)
                        print(f"Opened: {filepath.name}")

                        merged_frames.append(gdf)
                        added_from_zip = True
                        break  # Stop after successfully reading one file per zip
                    except Exception:
                        continue
                if added_from_zip:
                    break

        if not added_from_zip:
            zips_without_layer.append(zip_file.name)

    if zips_without_layer:
        raise RuntimeError(
            f"No readable geospatial layer found in {len(zips_without_layer)} archive(s) "
            f"under {zip_dir}: {', '.join(sorted(zips_without_layer))}"
        )
    if not merged_frames:
        raise RuntimeError(f"No zip archives to merge under {zip_dir}")

    merged_gdf = gpd.GeoDataFrame(
        pd.concat(merged_frames, ignore_index=True),
        geometry="geometry",
        crs=merged_frames[0].crs,
    )
    out_file = output_path / output_filename
    with atomic_output(out_file) as tmp_path:
        merged_gdf.to_file(tmp_path, driver="GPKG")
    print(f"\n✅ Merged GeoDataFrame written to {out_file}")

def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(
        description="Merge watershed zip archives into one GeoPackage per level."
    )
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Load config overrides and build the configured combined watershed layer."""
    args = parse_args()
    overrides = parse_config_overrides(args=args)

    if overrides.get("level") is not None:
        # An explicit --level must be honored, not discarded in favor of
        # discovering every lvl* directory - see combine_watersheds F1.
        levels = [str(overrides["level"])]
    else:
        cfg = load_config(script_name="combine_watersheds", **overrides)
        levels_root = os.path.dirname(cfg["paths"]["watersheds_zip_dir"])
        if not os.path.isdir(levels_root):
            raise FileNotFoundError(f"Watershed levels root not found: {levels_root}")
        levels = [
            k.replace('lvl', '') for k in os.listdir(levels_root)
            if k.startswith('lvl') and os.path.isdir(os.path.join(levels_root, k))
        ]

    for level in levels:
        # Reuse the parsed overrides wholesale so no field can be dropped here -
        # the hand-rolled dict this replaced silently discarded dynamic_buffering
        # and size_measure.
        cfg = load_config(
            script_name="combine_watersheds",
            **{**overrides, "level": level},
        )
        extract_and_merge_geodata(
            cfg["paths"]["watersheds_zip_dir"],
            os.path.dirname(cfg['paths']['watershed']),
            output_filename=os.path.basename(cfg['paths']['watershed'])
        )

if __name__ == '__main__':
    configure_logging()
    main()

