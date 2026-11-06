import argparse
import os
import shutil
import pandas as pd
import geopandas as gpd
from wwtp.starter import add_standard_override_arguments, load_config, parse_config_overrides
from wwtp.utils import configure_logging, ensure_output_dir_for_file


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run copy_falsy_images.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="copy_falsy_images", **overrides)
    gdf = gpd.read_file(cfg["paths"]["annotated_all_filepath"])
    if "category_number" not in gdf.columns:
        raise KeyError("Missing expected columns in corrected_all dataset: ['category_number']")

    image_col = None
    if "image_y" in gdf.columns:
        image_col = "image_y"
    elif "image" in gdf.columns:
        image_col = "image"
    else:
        raise KeyError("Missing expected columns in corrected_all dataset: ['image_y']")

    gdf["category_number"] = gdf["category_number"].fillna(-1).astype(int)
    falsy_gdf = gdf[
        ((gdf["category_number"] == -1) & gdf[image_col].notna())
    ].copy()  #| (gdf['category_number'].isin([8, 2, 3, 7, 1]))]
    image_input_dir = cfg["paths"]["annotated_images_output_dir"]
    falsy_gdf["filepath"] = falsy_gdf[image_col].apply(lambda x: os.path.join(image_input_dir, x))
    output_dir = os.path.join(os.path.dirname(cfg["paths"]["annotations_verf_image_outpath_dir"]), "falsy_images")
    ensure_output_dir_for_file(os.path.join(output_dir, "placeholder.txt"))

    for _, row in falsy_gdf.iterrows():
        src_path = row["filepath"]
        if pd.isna(src_path):
            continue

        filename = os.path.basename(src_path)
        dest_path = os.path.join(output_dir, filename)
        if os.path.exists(dest_path):
            continue

        try:
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
            else:
                print(f"File not found: {src_path}")
        except Exception as e:
            print(f"Error copying {filename}: {e}")


if __name__ == "__main__":
    configure_logging()
    main()
