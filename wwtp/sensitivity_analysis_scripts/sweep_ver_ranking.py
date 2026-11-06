"""Sweep verification ranking across the sweep grid of config.yaml.

The combinations are ``wwtp.workflow.sweep_variants()`` - the grid the
Snakemake ``sweep`` target builds - and each population layer is
``wwtp.workflow.population_layer``, the path add_pop writes.

Execution modes
---------------
Worker  (SLURM array task):
    python -m wwtp.sensitivity_analysis_scripts.sweep_ver_ranking TASK_ID [SHUFFLE_SEED]
    Processes ~1/10 of the sweep grid and writes raw per-subset metrics to
    ``{output_dir}/partial/task_{TASK_ID:02d}_metrics.csv``.

Merge (run after all 10 tasks finish):
    python -m wwtp.sensitivity_analysis_scripts.sweep_ver_ranking --merge
    Reads all partial CSVs, computes three levels of ranked CSV outputs.

Ranking hierarchy
-----------------
Subset    (alias, subset_type, source)          — most granular, 6 pairs per file
Subgroup  (alias, source)                       — ver+unver+single combined per source
Configuration (alias)                           — HW+EU combined, the primary ranking
"""

import argparse
import logging
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..utils import configure_logging, default_cpu_workers, reported_population_mask
from ..geo_utils import load_eu_reference_layer
from ..pop_validation_scripts.verification_script import find_verification_watersheds
from .. import workflow
from ..sensitivity_analysis_scripts.compare_pop_sweep_hw_eu import (
    get_latest_year_column,
    compute_sensitivity_metrics,
    _safe_zscore,
)
from ..data_merge.merge_seg_results import assign_to_nearest

logger = logging.getLogger(__name__)

NUM_TASKS = 10
DEFAULT_SEED = 42
SUBSET_TYPES = ("ver", "unver", "single")

# Columns written to every partial CSV — the merge step needs all of them.
_PARAM_COLS = [
    "alias",
    "level",
    "version",
    "buffer",
    "buffer_path_token",
    "weight_method",
    "weight_type",
    "weight_func",
    "weight_func_suffix",
    "dynamic_buffering",
    "size_measure",
    "approach",
]

# ── Process-local globals for parallel workers ────────────────────────────────

_REF_GDF = None
_THRESHOLD = None
_PERCENT_VERIFICATION = None


# ── Combo enumeration ─────────────────────────────────────────────────────────


def enumerate_sweep_records(approach, weight_funcs=None):
    """The expected population layer of ``approach`` for every combination of the sweep grid.

    Parameters
    ----------
    approach : str
        Voronoi approach key, e.g. ``"1"``.
    weight_funcs : list[str] | None
        Distance weightings to include; ``sweep.sweep_weight_funcs`` when None.

    Returns
    -------
    list[dict]
        One record per combination. Includes an ``exists`` flag; missing files
        are kept so callers can report pending combinations.
    """
    records = []
    for idx, variant in enumerate(workflow.sweep_variants(weight_funcs), start=1):
        cfg = workflow.config("add_pop", variant)
        filepath = workflow.population_layer(variant, approach)
        records.append(
            {
                "alias": f"F{idx:03d}",
                "filepath": filepath,
                "exists": os.path.isfile(filepath),
                "level": int(variant.level),
                "version": variant.version,
                "buffer": variant.buffer,
                "buffer_path_token": cfg["buffer_path_token"],
                "weight_method": variant.weight_method,
                "weight_type": cfg["weight_type"],
                "weight_func": variant.weight_func,
                "weight_func_suffix": cfg["weight_func_suffix"],
                "dynamic_buffering": str(variant.dynamic_buffering).lower(),
                "size_measure": variant.size_measure if variant.dynamic_buffering else "",
                "approach": approach,
            }
        )
    return records


def filter_by_task(records, task_id, num_tasks=NUM_TASKS, shuffle_seed=DEFAULT_SEED):
    """Return the subset of records assigned to ``task_id`` after seeded shuffle."""
    shuffled = list(records)
    random.Random(shuffle_seed).shuffle(shuffled)
    return [rec for idx, rec in enumerate(shuffled) if idx % num_tasks == task_id]


# ── Worker initialiser and per-record function ────────────────────────────────


def _init_ver_worker(eu_ref_filepath, factor, threshold, percent_verification):
    """Load EU reference into process-local state shared by all workers in the pool."""
    global _REF_GDF, _THRESHOLD, _PERCENT_VERIFICATION
    # This site used to drop the capacity multiplier that compare_pop_sweep_hw_eu
    # applied, so the two rankings would have diverged silently.
    _REF_GDF = load_eu_reference_layer(eu_ref_filepath, factor)
    _THRESHOLD = threshold
    _PERCENT_VERIFICATION = percent_verification


def _process_ver_record(record):
    """Compute raw HW/EU metric rows for ver/unver/single subsets of one record.

    Returns a list of dicts — one per (subset_type, source) combination.
    Returns an empty list when the file cannot be split or has no year column.
    """
    gdf = gpd.read_file(record["filepath"])
    year, year_col = get_latest_year_column(gdf)
    if year is None or year_col is None:
        logger.warning("No usable year column in %s — skipping", record["filepath"])
        return []

    try:
        gdf = find_verification_watersheds(gdf, _PERCENT_VERIFICATION)
    except (KeyError, ValueError) as exc:
        logger.warning("Cannot split %s into subsets: %s", record["filepath"], exc)
        return []

    subset_masks = {
        "ver": gdf["watersheds_chosen"],
        "unver": (~gdf["watersheds_chosen"]) & (~gdf["is_single_points"]),
        "single": gdf["is_single_points"],
    }

    rows = []
    for subset_type in SUBSET_TYPES:
        sub = gdf[subset_masks[subset_type]].reset_index(drop=True)
        if sub.empty:
            continue

        # HW: plants with reported population only (QUAL_POP == 1), as in hw_comparison.py
        hw_sub = sub[reported_population_mask(sub)].copy()
        hw_metrics = compute_sensitivity_metrics(hw_sub, year_col, "POP_SERVED")

        # EU: nearest-neighbour join, one source row per EU plant, matched rows only
        eu_sub = assign_to_nearest(sub.copy(), _REF_GDF, _THRESHOLD, one_to_one=True)
        if "uwwCapacity" in eu_sub.columns:
            eu_sub = eu_sub[eu_sub["uwwCapacity"].notna()].reset_index(drop=True)
        else:
            eu_sub = pd.DataFrame()
        eu_metrics = compute_sensitivity_metrics(eu_sub, year_col, "POP_SERVED_EU")

        for source, metric_block in (("HW", hw_metrics), ("EU", eu_metrics)):
            row = {k: record[k] for k in _PARAM_COLS}
            row["filepath"] = record["filepath"]
            row["subset_type"] = subset_type
            row["source"] = source
            row["year"] = year
            row.update(metric_block)
            rows.append(row)

    return rows


# ── Raw metrics collection ────────────────────────────────────────────────────


def collect_metrics(records, eu_ref_filepath, factor, threshold, percent_verification, max_workers=None):
    """Process all records in parallel and return a DataFrame of raw metric rows."""
    if max_workers is None:
        max_workers = default_cpu_workers()

    rows = []
    total = len(records)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_ver_worker,
        initargs=(eu_ref_filepath, factor, threshold, percent_verification),
    ) as executor:
        futures = {executor.submit(_process_ver_record, rec): rec for rec in records}
        for i, future in enumerate(as_completed(futures), start=1):
            rec = futures[future]
            logger.info("[%d/%d] %s", i, total, os.path.basename(rec["filepath"]))
            try:
                rows.extend(future.result())
            except Exception as err:
                logger.warning("Failed %s: %s", rec["filepath"], err)

    return pd.DataFrame(rows)


# ── Scoring ───────────────────────────────────────────────────────────────────


def compute_subset_scores(df):
    """Add ``subset_score`` by z-scoring error metrics within (subset_type, source).

    Lower score = lower bias/spread/log-RMSE = better.
    """
    df = df.copy()
    group_key = ["subset_type", "source"]
    for metric in ("median_abs_ndi", "iqr_ndi", "rmse_log_ratio"):
        df[f"z_{metric}"] = df.groupby(group_key)[metric].transform(_safe_zscore)
    df["subset_score"] = df[["z_median_abs_ndi", "z_iqr_ndi", "z_rmse_log_ratio"]].mean(axis=1)
    return df


def compute_subgroup_scores(subset_df):
    """Aggregate subset rows to (alias, source) by n-weighted mean of error metrics.

    Z-scores the aggregated metrics within each source group to produce
    ``subgroup_score``.  Also carries the parameter columns through.
    """
    error_metrics = ["median_abs_ndi", "iqr_ndi", "rmse_log_ratio", "spearman_r", "median_ndi", "median_alpha"]

    # Param cols that are NOT groupby keys — alias and source come from reset_index().
    non_key_params = [c for c in _PARAM_COLS if c != "alias"]

    def _wavg(group):
        weights = group["n"].clip(lower=0)
        total_n = weights.sum()
        result = {"n_total": int(total_n)}
        for m in error_metrics:
            if total_n > 0 and m in group.columns:
                result[m] = float(np.average(group[m].fillna(0), weights=weights))
            else:
                result[m] = np.nan
        # Carry non-key param columns (values are identical within an alias group)
        for col in non_key_params:
            if col in group.columns:
                result[col] = group[col].iloc[0]
        return pd.Series(result)

    subgroup = subset_df.groupby(["alias", "source"]).apply(_wavg, include_groups=False).reset_index()
    for metric in ("median_abs_ndi", "iqr_ndi", "rmse_log_ratio"):
        subgroup[f"z_{metric}"] = subgroup.groupby("source")[metric].transform(_safe_zscore)
    subgroup["subgroup_score"] = subgroup[["z_median_abs_ndi", "z_iqr_ndi", "z_rmse_log_ratio"]].mean(axis=1)
    return subgroup


def compute_config_scores(subgroup_df, hw_weight):
    """Combine HW and EU subgroup scores per alias into a single ``config_score``."""
    eu_weight = 1.0 - hw_weight
    pivot = subgroup_df.pivot_table(index="alias", columns="source", values="subgroup_score", aggfunc="mean")
    hw_s = pivot["HW"] if "HW" in pivot.columns else pd.Series(np.nan, index=pivot.index)
    eu_s = pivot["EU"] if "EU" in pivot.columns else pd.Series(np.nan, index=pivot.index)

    avail = (~hw_s.isna()).astype(float) * hw_weight + (~eu_s.isna()).astype(float) * eu_weight
    pivot["config_score"] = (hw_s.fillna(0) * hw_weight + eu_s.fillna(0) * eu_weight) / avail.replace(0, np.nan)
    pivot = pivot.rename(columns={"HW": "subgroup_score_HW", "EU": "subgroup_score_EU"})
    pivot["hw_weight"] = hw_weight
    pivot["eu_weight"] = eu_weight

    # _PARAM_COLS[0] is "alias" which is already the index — exclude it to avoid
    # duplicating the column before set_index, which produces all-NaN after join.
    non_alias_params = [c for c in _PARAM_COLS if c != "alias"]
    params = subgroup_df.drop_duplicates("alias")[["alias"] + non_alias_params].set_index("alias")
    config = pivot.join(params, how="left").reset_index()

    # Put meaningful param columns first so the CSV is readable without alias codes.
    score_cols = ["config_score", "subgroup_score_HW", "subgroup_score_EU", "hw_weight", "eu_weight"]
    ordered_cols = non_alias_params + score_cols + ["alias"]
    ordered_cols = [c for c in ordered_cols if c in config.columns]
    config = config[ordered_cols]

    return config.sort_values("config_score", ascending=True)


# ── Output ────────────────────────────────────────────────────────────────────


def _add_rank(df, sort_col):
    df = df.sort_values(sort_col, ascending=True).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def save_rankings(subset_df, subgroup_df, config_df, out_dir):
    """Write all three levels of ranking CSVs to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)

    # ── Subset rankings (one file per subset_type × source) ──────────────────
    for subset_type in SUBSET_TYPES:
        for source in ("HW", "EU"):
            sub = subset_df[(subset_df["subset_type"] == subset_type) & (subset_df["source"] == source)].copy()
            if sub.empty:
                continue
            sub = _add_rank(sub, "subset_score")
            path = os.path.join(out_dir, f"ranking_subset_{subset_type}_{source}.csv")
            sub.to_csv(path, index=False)
            logger.info("Saved %s", path)

    # ── Subgroup rankings (one file per source) ───────────────────────────────
    for source in ("HW", "EU"):
        sub = subgroup_df[subgroup_df["source"] == source].copy()
        if sub.empty:
            continue
        sub = _add_rank(sub, "subgroup_score")
        path = os.path.join(out_dir, f"ranking_subgroup_{source}.csv")
        sub.to_csv(path, index=False)
        logger.info("Saved %s", path)

    # ── Configuration ranking (primary output) ────────────────────────────────
    config_df = _add_rank(config_df, "config_score")
    path = os.path.join(out_dir, "ranking_configurations.csv")
    config_df.to_csv(path, index=False)
    logger.info("Saved %s", path)


# ── CLI modes ─────────────────────────────────────────────────────────────────


def _load_script_cfg(overrides=None):
    """Return the sweep_ver_ranking section of config.yaml."""
    return load_config(script_name="sweep_ver_ranking", **(overrides or {}))


def main_worker(task_id, shuffle_seed, overrides=None):
    """Process this task's share of the sweep grid and write a partial metrics CSV."""
    cfg = _load_script_cfg(overrides)
    eu_ref_filepath = cfg["paths"]["eu_ref_filepath"]
    out_dir = cfg["paths"]["output_dir"]
    threshold = cfg["threshold"]
    factor = cfg["eu_reference_factor"]
    percent_verification = cfg["percent_verification"]
    max_workers = cfg["max_workers"]
    approach = str(cfg["figures"]["approach"])

    partial_dir = os.path.join(out_dir, "partial")
    os.makedirs(partial_dir, exist_ok=True)

    logger.info("Task %d/%d (seed=%d): enumerating sweep grid …", task_id, NUM_TASKS - 1, shuffle_seed)
    all_records = enumerate_sweep_records(approach=approach)

    # Write the full combination map once (all tasks write the same content, safe to overwrite)
    combo_path = os.path.join(out_dir, "sweep_combinations.csv")
    pd.DataFrame(all_records).to_csv(combo_path, index=False)

    assigned = filter_by_task(all_records, task_id, num_tasks=NUM_TASKS, shuffle_seed=shuffle_seed)
    existing = [r for r in assigned if r["exists"]]
    pending = len(assigned) - len(existing)

    logger.info(
        "Task %d: %d assigned combos — %d with output files, %d pending.",
        task_id,
        len(assigned),
        len(existing),
        pending,
    )
    if not existing:
        logger.warning("Task %d: no output files found — nothing to process.", task_id)
        return

    metrics_df = collect_metrics(
        existing,
        eu_ref_filepath,
        factor,
        threshold,
        percent_verification,
        max_workers=max_workers,
    )
    if metrics_df.empty:
        logger.warning("Task %d: no metrics generated.", task_id)
        return

    out_path = os.path.join(partial_dir, f"task_{task_id:02d}_metrics.csv")
    metrics_df.to_csv(out_path, index=False)
    logger.info("Task %d: wrote %d rows → %s", task_id, len(metrics_df), out_path)


def main_merge(overrides=None):
    """Read all partial CSVs, compute three-level scores, and write ranking CSVs."""
    cfg = _load_script_cfg(overrides)
    out_dir = cfg["paths"]["output_dir"]
    hw_weight = float(cfg["hw_weight"])

    partial_dir = os.path.join(out_dir, "partial")
    partial_files = sorted(glob(os.path.join(partial_dir, "task_*_metrics.csv")))
    if not partial_files:
        logger.error("No partial metric files found in %s — run worker tasks first.", partial_dir)
        sys.exit(1)

    logger.info("Merging %d partial file(s) from %s …", len(partial_files), partial_dir)
    frames = [pd.read_csv(f) for f in partial_files]
    full_df = pd.concat(frames, ignore_index=True)
    full_df.to_csv(os.path.join(out_dir, "sweep_ver_ranking_full_metrics.csv"), index=False)
    logger.info("Full metrics: %d rows across %d files.", len(full_df), full_df["alias"].nunique())

    # Three-level scoring
    subset_df = compute_subset_scores(full_df)
    subgroup_df = compute_subgroup_scores(subset_df)
    config_df = compute_config_scores(subgroup_df, hw_weight)

    save_rankings(subset_df, subgroup_df, config_df, out_dir)
    logger.info("Merge complete. Rankings written to %s", out_dir)


def main():
    """Dispatch to worker or merge mode based on CLI arguments."""
    parser = argparse.ArgumentParser(description="Sweep verification ranking: worker or merge mode.")
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        metavar="TASK_ID",
        help="SLURM array task ID (0-%d); runs worker mode." % (NUM_TASKS - 1),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all partial CSVs and compute final rankings.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="SHUFFLE_SEED",
        help="Random seed for combo shuffling (default: %d)." % DEFAULT_SEED,
    )
    add_standard_override_arguments(parser)

    args = parser.parse_args()
    overrides = parse_config_overrides(args=args)

    if args.merge:
        main_merge(overrides)
    else:
        if args.task_id is None:
            parser.error("--task-id is required when --merge is not provided.")
        if args.task_id < 0 or args.task_id >= NUM_TASKS:
            parser.error(f"--task-id must be between 0 and {NUM_TASKS - 1}.")
        main_worker(args.task_id, args.shuffle_seed, overrides)


if __name__ == "__main__":
    configure_logging()
    main()
