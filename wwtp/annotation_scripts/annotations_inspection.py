"""Create a stratified annotation review sample and inspection artifacts.

This script reads annotation model outputs, parses category labels, plots class
distribution, and copies sampled images into per-category folders for manual QA.
"""

import argparse
import os
import re
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from wwtp.annotation_scripts.merge_annotations import decode_gen_text
from wwtp.utils import atomic_output, configure_logging, ensure_output_dir_for_file
from wwtp.starter import add_standard_override_arguments, load_config, parse_config_overrides


def plot_category_distribution(df, column="category_name", save_path=None, show=False):
    """Plot category frequency (including missing values) as a bar chart."""
    counts = df[column].fillna("NaN").value_counts()

    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar", color="skyblue", edgecolor="black")

    plt.title(f"Distribution of {column}", fontsize=14)
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        ensure_output_dir_for_file(save_path)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def get_stratified_sample(df, target_column, total_n=None, seed=None):
    """Return near-balanced sample by category, with remainder backfill.

    Missing categories are treated as an explicit ``NaN`` class.
    """
    if total_n is None or seed is None:
        raise ValueError("total_n and seed must be provided")

    temp_df = df.copy()
    temp_df["temp_grp"] = temp_df[target_column].fillna("NaN")

    unique_cats = temp_df["temp_grp"].unique()
    if len(unique_cats) == 0:
        return temp_df.drop(columns=["temp_grp"])

    n_per_cat = total_n // len(unique_cats)

    sampled = temp_df.groupby("temp_grp", group_keys=False).apply(
        lambda x: x.sample(min(len(x), n_per_cat), random_state=seed)
    )

    if len(sampled) < total_n:
        shortfall = total_n - len(sampled)
        remaining_pool = temp_df.drop(sampled.index)

        extra = remaining_pool.sample(min(len(remaining_pool), shortfall), random_state=seed)
        sampled = pd.concat([sampled, extra])

    return sampled.drop(columns=["temp_grp"], errors="ignore").reset_index(drop=True)


def sanitize_folder_name(name):
    """Normalize folder names to safe filesystem tokens."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", str(name).strip())
    return cleaned if cleaned else "Uncategorized"


def organize_files_by_category(df, source_col, category_col, base_dir):
    """Create one folder per category and copy sampled images into it."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created base directory: {base_dir}")

    copy_count = 0
    error_count = 0

    for _, row in df.iterrows():
        cat_name = str(row[category_col]).strip() if pd.notna(row[category_col]) else "Uncategorized"
        cat_name = sanitize_folder_name(cat_name)
        src_path = row[source_col]

        if pd.isna(src_path):
            continue

        target_folder = os.path.join(base_dir, cat_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        filename = os.path.basename(src_path)
        dest_path = os.path.join(target_folder, filename)

        try:
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
                copy_count += 1
            else:
                print(f"File not found: {src_path}")
                error_count += 1
        except Exception as e:
            print(f"Error copying {filename}: {e}")
            error_count += 1

    print(f"Done! Files copied: {copy_count} | Errors: {error_count}")


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run annotations_inspection.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Run full inspection flow: parse, plot, sample, export, and copy images."""
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="annotations_inspection", **overrides)

    image_input_dir = cfg["paths"]["annotated_images_output_dir"]
    image_output_dir = cfg["paths"]["annotations_verf_image_outpath_dir"]
    filepath = cfg["paths"]["annotations_results_filepath"]
    os.makedirs(image_output_dir, exist_ok=True)

    df = pd.read_csv(filepath)
    required_cols = {"gen_text", "image"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise KeyError(f"Missing expected annotation columns: {sorted(missing_cols)}")

    df[["category_number", "category_name", "justification"]] = df["gen_text"].apply(
        lambda x: pd.Series(decode_gen_text(x))
    )
    df["filepath"] = df["image"].apply(lambda x: os.path.join(image_input_dir, x))

    sample_size = int(cfg["annotations"]["n_sample_size"])
    if sample_size < 1:
        raise ValueError("annotations.n_sample_size must be >= 1")
    random_seed = int(cfg["annotations"]["random_seed"])

    histogram_path = os.path.join(image_output_dir, "annotation_category_histogram.png")
    plot_category_distribution(df, "category_name", save_path=histogram_path, show=False)

    df_sampled = get_stratified_sample(df, "category_name", total_n=sample_size, seed=random_seed)

    sample_csv_path = os.path.join(image_output_dir, "annotation_stratified_sample.csv")
    with atomic_output(sample_csv_path) as tmp_path:
        df_sampled.to_csv(tmp_path, index=False)
    print(f"Saved sampled rows to {sample_csv_path}")

    print(f"New shape: {df_sampled.shape}")
    print(df_sampled["category_name"].value_counts(dropna=False))
    organize_files_by_category(df_sampled, "filepath", "category_name", image_output_dir)


if __name__ == "__main__":
    configure_logging()
    main()
