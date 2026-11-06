"""Sample N image filenames per annotation class from model output CSV.

Reads the raw inference CSV, parses category numbers from the ``gen_text``
column using the same logic as ``merge_annotations``, then randomly samples
up to N filenames per class.  The result is written as a JSON array of arrays
(one inner list per class, ordered by category number).
"""

import argparse
import json
import re

import pandas as pd

from wwtp.starter import add_standard_override_arguments, load_config, parse_config_overrides
from wwtp.utils import configure_logging, ensure_output_dir_for_file


def _decode_category_number(text: str) -> str | None:
    """Return the category number string from a raw model output cell."""
    if not isinstance(text, str):
        return None
    match = re.search(r"Decision:\s*(\d+)\.", text)
    return match.group(1).strip() if match else None


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run sample_annotations_by_class.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="sample_annotations_by_class", **overrides)

    n_per_class: int = cfg["sampling"]["n_per_class"]
    random_seed: int = cfg["annotations"]["random_seed"]
    filepath: str = cfg["paths"]["annotations_results_filepath"]
    output_path: str = cfg["paths"]["sampled_images_output_filepath"]

    df = pd.read_csv(filepath)
    required_cols = {"gen_text", "image"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise KeyError(f"Missing expected annotation columns: {sorted(missing_cols)}")

    df["category_number"] = df["gen_text"].apply(_decode_category_number)
    df = df[df["category_number"].notna() & df["image"].notna()].copy()

    sampled: list[list[str]] = []
    for category_number, group in sorted(df.groupby("category_number"), key=lambda x: int(x[0])):
        images = group["image"].tolist()
        n = min(n_per_class, len(images))
        sample = group["image"].sample(n=n, random_state=random_seed).tolist()
        sampled.append(sample)
        print(f"Class {category_number}: {len(images)} total, sampled {n}")

    ensure_output_dir_for_file(output_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(sampled, fh, indent=2)
    print(f"Wrote {len(sampled)} classes to {output_path}")


if __name__ == "__main__":
    configure_logging()
    main()
