"""Merge annotation model outputs back into the main WWTP dataset.

The script parses model text responses into structured fields and joins them to
the main geospatial table via the numeric identifier encoded in image names.
"""

import argparse
import os
import re

import geopandas as gpd
import pandas as pd

from wwtp.starter import add_standard_override_arguments, load_config, parse_config_overrides
from wwtp.utils import atomic_output, configure_logging

def decode_gen_text(text):
    """Parse model output text into category number/name and justification.

    Expected format is:
    "Analysis: ...\\n\\nDecision: <number>. <name>\\nJustification: <justification>"

    The parser is tolerant to malformed rows and returns ``None`` values when a
    reliable parse is not possible.
    """
    if not isinstance(text, str):
        return None, None, None

    category_number = None
    category_name = None
    justification = None

    decision_match = re.search(r"Decision:\s*(\d+)\.\s*(.+?)(?:\n|$)", text)
    if decision_match:
        category_number = decision_match.group(1).strip()
        category_name = decision_match.group(2).strip()

    justification_match = re.search(r"Justification:\s*(.+)", text, re.DOTALL)
    if justification_match:
        justification = justification_match.group(1).strip()

    def _clean_field(value):
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value or value.startswith("["):
            return None
        return value

    return _clean_field(category_number), _clean_field(category_name), _clean_field(justification)


def parse_idx_from_image_name(image_name):
    """Extract numeric ``idx`` from image filenames like ``tile_123.png``."""
    if not isinstance(image_name, str):
        return None

    stem = os.path.splitext(os.path.basename(image_name))[0]
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else None

def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run merge_annotations.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Load annotations, merge parsed labels onto WWTP points, and overwrite output.

    Returns
    -------
    None
        The merged geospatial output is written back to the configured dataset.
    """
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="merge_annotations", **overrides)

    image_input_dir = cfg['paths']['annotated_images_output_dir']
    filepath = cfg['paths']['annotations_results_filepath']

    df = pd.read_csv(filepath)
    required_cols = {'gen_text', 'image'}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise KeyError(f"Missing expected annotation columns: {sorted(missing_cols)}")

    df[['category_number', 'category_name', 'justification']] = df['gen_text'].apply(
        lambda x: pd.Series(decode_gen_text(x))
    )
    df['filepath'] = df['image'].apply(lambda x: os.path.join(image_input_dir, x))

    points_df = gpd.read_file(cfg['paths']['corrected_all_filepath'])
    if 'idx' not in points_df.columns:
        raise KeyError("Missing required 'idx' column in corrected_all dataset")
    points_df['idx'] = points_df['idx'].astype(int)

    annotation_cols = ['category_number', 'category_name', 'justification']

    df['idx'] = df['image'].apply(parse_idx_from_image_name)
    df = df[df['idx'].notna()].copy()
    df['idx'] = df['idx'].astype(int)
    df.drop(columns=['filepath', 'gen_text'], inplace=True)

    # Merge parsed annotations onto the geospatial points table.
    # Existing annotation values in points_df are kept unless overridden by df.
    merged_df = gpd.GeoDataFrame(pd.merge(points_df, df, on='idx', how='left'), geometry='geometry', crs=points_df.crs)
    for col in annotation_cols:
        col_x, col_y = f"{col}_x", f"{col}_y"
        if col_x in merged_df.columns and col_y in merged_df.columns:
            merged_df[col] = merged_df[col_y].where(merged_df[col_y].notna(), merged_df[col_x])
            merged_df.drop(columns=[col_x, col_y], inplace=True)
    # Write to a NEW path, never back onto corrected_all_filepath: that was a
    # self-loop (read and overwrite the same file) that destroyed the
    # distinction between annotated and unannotated data on every re-run.
    with atomic_output(cfg['paths']['annotated_all_filepath']) as tmp_path:
        merged_df.to_file(tmp_path, index=False, driver='GPKG')

if __name__ == "__main__":
    configure_logging()
    main()  


