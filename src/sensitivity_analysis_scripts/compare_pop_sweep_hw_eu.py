"""Sensitivity analysis across all population-enriched Voronoi outputs.

This script is intentionally single-run and single-year-per-file: for each
GPKG it selects the latest available *_zonal_sum year and computes robust
error metrics against both HydroWaste (HW) and EU references.

The resulting dashboards are split left/right in one figure:
- left panel: HW metrics
- right panel: EU metrics

Files are shown as aliases (F001, F002, ...). The alias mapping is written to
CSV outside the plots so charts remain readable when many files are compared.
"""

import argparse
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..utils import configure_logging, default_cpu_workers, ensure_output_dir_for_file, reported_population_mask, resolve_latest_zonal_sum_column
from ..geo_utils import load_eu_reference_layer
from ..pop_validation_scripts.hw_comparison import ndvi, multiples, replace_inf
from ..data_merge.merge_seg_results import assign_to_nearest


logger = logging.getLogger(__name__)


_REF_GDF = None
_THRESHOLD = None


def parse_pop_output_path(filepath):
    """Extract sensitivity parameters encoded in a pop-output filepath."""
    normalized = str(filepath).replace("\\", "/")
    # Buffer token: fixed radius in metres (bf2500), the size measure of the
    # service radius (bfrep, bffp), or the forms older outputs carry (bfq0_6, bfk0_75).
    buffer_token_pattern = r"(?:\d+|[kq]\d+(?:[._]\d+)*|rep|fp)"
    path_pattern = (
        r"/pop_voronoi_layers/v(?P<version>[^/]+)/lvl(?P<level>\d+)/"
        rf"bf(?P<buffer>{buffer_token_pattern})(?:/(?P<weight_type>[^/]+))?/(?P<filename>[^/]+)$"
    )
    path_match = re.search(path_pattern, normalized)
    if not path_match:
        return None

    filename = path_match.group("filename")
    file_pattern = (
        r"^(?:pop_added_)?appr_(?P<approach>[012])(?P<only_round>_only_round)?"
        rf"_v(?P<file_version>[^_]+)_lvl(?P<file_level>\d+)_bf(?P<file_buffer>{buffer_token_pattern})"
        r"(?P<weight_func>.*?)\.gpkg$"
    )
    file_match = re.match(file_pattern, filename)
    if not file_match:
        return None

    # Guard against malformed paths where directory and filename disagree.
    if file_match.group("file_buffer") != path_match.group("buffer"):
        return None

    return {
        "filepath": normalized,
        "filename": filename,
        "version": path_match.group("version"),
        "level": int(path_match.group("level")),
        "buffer": path_match.group("buffer"),
        "weight_type": path_match.group("weight_type") or "",
        "approach": file_match.group("approach"),
        "only_round": bool(file_match.group("only_round")),
        "weight_func": file_match.group("weight_func"),
    }


def list_pop_output_files(data_dir):
    """List all parseable pop-output GPKGs in the sweep root.

    When ``weight_func`` is empty the distance weighting is disabled, so
    ``weight_method`` (and therefore ``weight_type``) has no effect on the
    output. Any such duplicates that differ only in ``weight_type`` are
    deduplicated here: only the first encountered file per
    ``(version, level, buffer, approach, only_round)`` group with an empty
    ``weight_func`` is kept.
    """
    root = Path(data_dir) / "pop_voronoi_layers"
    if not root.exists():
        logger.warning("pop_voronoi_layers directory not found: %s", root)
        return []

    records = []
    parse_failures = []
    seen_empty_wf = set()
    for filepath in sorted(root.rglob("*.gpkg")):
        if Path(filepath).name.startswith('temp_'):
            continue

        params = parse_pop_output_path(filepath)
        if params is None:
            parse_failures.append(str(filepath).replace("\\", "/"))
            continue

        if params["weight_func"] == "":
            key = (params["version"], params["level"], params["buffer"],
                   params["approach"], params["only_round"])
            if key in seen_empty_wf:
                logger.debug(
                    "Skipping duplicate empty-weight_func file (weight_type=%s differs but output is identical): %s",
                    params["weight_type"], filepath,
                )
                continue
            seen_empty_wf.add(key)

        records.append(params)

    if not records and parse_failures:
        logger.warning(
            "Found %d GPKG files under %s but none matched expected naming/path patterns. "
            "Examples: %s",
            len(parse_failures),
            root,
            "; ".join(parse_failures[:5]),
        )
    return records


def make_aliases(records):
    """Attach stable aliases (F001, F002, ...) to each file record."""
    for idx, record in enumerate(records, start=1):
        record["alias"] = f"F{idx:03d}"
    return records


def get_latest_year_column(gdf):
    """Return (year, column_name) for the latest available *_zonal_sum column.

    2014 comes from a different WorldPop release than the yearly series, so it is
    excluded here even though the figure scripts accept it.
    """
    return resolve_latest_zonal_sum_column(
        gdf,
        exclude_years=(2014,),
        keep_unparseable=False,
        required=False,
    )


def compute_sensitivity_metrics(df, prediction_col, reference_col):
    """Compute robust sensitivity metrics from paired prediction/reference values."""
    if prediction_col not in df.columns or reference_col not in df.columns:
        return {
            "n": 0,
            "mean_ndi": np.nan,
            "median_ndi": np.nan,
            "std_ndi": np.nan,
            "median_abs_ndi": np.nan,
            "iqr_ndi": np.nan,
            "mean_alpha": np.nan,
            "median_alpha": np.nan,
            "std_alpha": np.nan,
            "rmse_log_ratio": np.nan,
            "spearman_r": np.nan,
        }

    work = df[[prediction_col, reference_col]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return {
            "n": 0,
            "mean_ndi": np.nan,
            "median_ndi": np.nan,
            "std_ndi": np.nan,
            "median_abs_ndi": np.nan,
            "iqr_ndi": np.nan,
            "mean_alpha": np.nan,
            "median_alpha": np.nan,
            "std_alpha": np.nan,
            "rmse_log_ratio": np.nan,
            "spearman_r": np.nan,
        }

    eps = 1e-3
    work = ndvi(work, prediction_col, reference_col, "tmp_ndi", small_value=eps)
    work = multiples(work, prediction_col, reference_col, "tmp_alpha", small_value=eps)
    work = replace_inf(work, "tmp_ndi")
    work = replace_inf(work, "tmp_alpha")
    work = work.dropna(subset=["tmp_ndi", "tmp_alpha"]).copy()

    if work.empty:
        return {
            "n": 0,
            "mean_ndi": np.nan,
            "median_ndi": np.nan,
            "std_ndi": np.nan,
            "median_abs_ndi": np.nan,
            "iqr_ndi": np.nan,
            "mean_alpha": np.nan,
            "median_alpha": np.nan,
            "std_alpha": np.nan,
            "rmse_log_ratio": np.nan,
            "spearman_r": np.nan,
        }

    ndi = work["tmp_ndi"]
    alpha = work["tmp_alpha"]

    pred = work[prediction_col].clip(lower=0)
    ref = work[reference_col].clip(lower=0)
    log_ratio = np.log((pred + eps) / (ref + eps))
    rmse_log_ratio = float(np.sqrt(np.nanmean(log_ratio ** 2))) if len(log_ratio) else np.nan
    spearman_r = float(work[prediction_col].corr(work[reference_col], method="spearman"))

    return {
        "n": int(len(work)),
        "mean_ndi": float(ndi.mean()),
        "median_ndi": float(ndi.median()),
        "std_ndi": float(ndi.std(ddof=0)),
        "median_abs_ndi": float(np.abs(ndi).median()),
        "iqr_ndi": float(ndi.quantile(0.75) - ndi.quantile(0.25)),
        "mean_alpha": float(alpha.mean()),
        "median_alpha": float(alpha.median()),
        "std_alpha": float(alpha.std(ddof=0)),
        "rmse_log_ratio": rmse_log_ratio,
        "spearman_r": spearman_r,
    }


def _init_summary_worker(eu_ref_filepath, factor, threshold):
    """Initialize process-local references for per-file metric computation."""
    global _REF_GDF, _THRESHOLD
    _REF_GDF = load_eu_reference_layer(eu_ref_filepath, factor)
    _THRESHOLD = threshold


def _process_single_record(record):
    """Compute HW/EU metric rows for one pop-output record."""
    gdf = gpd.read_file(record["filepath"])

    year, year_col = get_latest_year_column(gdf)
    if year is None or year_col is None:
        return []

    hw_subset = gdf[reported_population_mask(gdf)].copy()
    hw_metrics = compute_sensitivity_metrics(hw_subset, year_col, "POP_SERVED")

    eu_subset = assign_to_nearest(gdf.copy(), _REF_GDF, _THRESHOLD, one_to_one=True)
    if "uwwCapacity" in eu_subset.columns:
        eu_subset = eu_subset[eu_subset["uwwCapacity"].notna()].reset_index(drop=True)
    eu_metrics = compute_sensitivity_metrics(eu_subset, year_col, "POP_SERVED_EU")

    rows = []
    for source, metric_block in (("HW", hw_metrics), ("EU", eu_metrics)):
        row = {
            "alias": record["alias"],
            "filepath": record["filepath"],
            "filename": record["filename"],
            "version": record["version"],
            "level": record["level"],
            "buffer": record["buffer"],
            "weight_type": record["weight_type"],
            "weight_func": record["weight_func"],
            "approach": record["approach"],
            "only_round": record["only_round"],
            "source": source,
            "year": year,
        }
        row.update(metric_block)
        rows.append(row)
    return rows


def build_summary_table(records, eu_ref_filepath, threshold, factor, max_workers=None):
    """Compute one-row-per-file-per-source summary using parallel file processing."""
    rows = []
    total = len(records)

    if max_workers is None:
        max_workers = default_cpu_workers()

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_summary_worker,
        initargs=(eu_ref_filepath, factor, threshold),
    ) as executor:
        futures = {executor.submit(_process_single_record, record): record for record in records}
        for i, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            logger.info("[%d/%d] Processing %s", i, total, record["filename"])
            try:
                rows.extend(future.result())
            except Exception as err:
                logger.warning("Failed to process %s: %s", record["filepath"], err)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    for metric in ["median_abs_ndi", "iqr_ndi", "rmse_log_ratio"]:
        z_name = f"z_{metric}"
        summary[z_name] = summary.groupby("source")[metric].transform(_safe_zscore)

    # Lower score means lower bias/error/instability.
    summary["sensitivity_score"] = summary[
        ["z_median_abs_ndi", "z_iqr_ndi", "z_rmse_log_ratio"]
    ].mean(axis=1)

    summary["hw_rank"] = summary[summary["source"] == "HW"]["sensitivity_score"].rank(method="dense")
    summary["eu_rank"] = summary[summary["source"] == "EU"]["sensitivity_score"].rank(method="dense")
    return summary


def _safe_zscore(series):
    """Return z-score while handling all-NaN or zero-variance vectors."""
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean(skipna=True)
    std = s.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mean) / std


def build_alias_order(summary, hw_weight, eu_weight=None):
    """Create a single alias ordering from weighted HW/EU scores."""
    if eu_weight is None:
        eu_weight = 1 - hw_weight

    if hw_weight < 0 or eu_weight < 0:
        raise ValueError("hw_weight and eu_weight must be non-negative")

    total_weight = hw_weight + eu_weight
    if total_weight == 0:
        raise ValueError("At least one of hw_weight or eu_weight must be > 0")

    hw_w = hw_weight / total_weight
    eu_w = eu_weight / total_weight

    score_pivot = summary.pivot_table(index="alias", columns="source", values="sensitivity_score", aggfunc="mean")
    hw_scores = score_pivot["HW"] if "HW" in score_pivot.columns else pd.Series(np.nan, index=score_pivot.index)
    eu_scores = score_pivot["EU"] if "EU" in score_pivot.columns else pd.Series(np.nan, index=score_pivot.index)

    weighted_sum = hw_scores.fillna(0) * hw_w + eu_scores.fillna(0) * eu_w
    available_weight = (~hw_scores.isna()).astype(float) * hw_w + (~eu_scores.isna()).astype(float) * eu_w

    score_pivot["joint_score"] = weighted_sum / available_weight.replace(0, np.nan)
    score_pivot["hw_weight_used"] = hw_w
    score_pivot["eu_weight_used"] = eu_w
    score_pivot = score_pivot.sort_values("joint_score", ascending=True)
    ordered_aliases = score_pivot.index.tolist()
    return ordered_aliases, score_pivot.reset_index()


def plot_split_score_bars(summary, ordered_aliases, output_filepath):
    """Plot split chart with HW scores on the left and EU scores on the right."""
    hw = summary[summary["source"] == "HW"].set_index("alias").reindex(ordered_aliases)
    eu = summary[summary["source"] == "EU"].set_index("alias").reindex(ordered_aliases)

    y = np.arange(len(ordered_aliases))
    fig_height = max(10, min(44, 0.18 * len(ordered_aliases) + 4))
    fig, axes = plt.subplots(1, 2, figsize=(18, fig_height), sharey=True)

    axes[0].barh(y, hw["sensitivity_score"], color="#4C78A8", alpha=0.85)
    axes[0].set_title("HW Sensitivity Score (lower is better)")
    axes[0].set_xlabel("Composite score")
    axes[0].set_ylabel("File alias")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ordered_aliases, fontsize=7)
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(y, eu["sensitivity_score"], color="#F58518", alpha=0.85)
    axes[1].set_title("EU Sensitivity Score (lower is better)")
    axes[1].set_xlabel("Composite score")
    axes[1].set_ylabel("")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].grid(axis="x", alpha=0.25)

    axes[0].invert_yaxis()
    fig.suptitle("Sensitivity Ranking Dashboard: HW (left) vs EU (right)", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    ensure_output_dir_for_file(output_filepath)
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)


def _to_percentile_goodness(series, lower_is_better=True):
    """Convert metric values into 0..1 percentile goodness scores."""
    s = pd.to_numeric(series, errors="coerce")
    rank = s.rank(method="average", pct=True, na_option="keep")
    if lower_is_better:
        return 1 - rank
    return rank


def plot_split_metric_profiles(summary, ordered_aliases, output_filepath):
    """Plot split HW/EU heatmaps with normalized metric goodness profiles."""
    work = summary.copy()

    work["good_abs_ndi"] = work.groupby("source")["median_abs_ndi"].transform(
        lambda s: _to_percentile_goodness(s, lower_is_better=True)
    )
    work["good_iqr_ndi"] = work.groupby("source")["iqr_ndi"].transform(
        lambda s: _to_percentile_goodness(s, lower_is_better=True)
    )
    work["good_rmse_log_ratio"] = work.groupby("source")["rmse_log_ratio"].transform(
        lambda s: _to_percentile_goodness(s, lower_is_better=True)
    )
    work["good_spearman"] = work.groupby("source")["spearman_r"].transform(
        lambda s: _to_percentile_goodness(s, lower_is_better=False)
    )
    work["good_n"] = work.groupby("source")["n"].transform(
        lambda s: _to_percentile_goodness(s, lower_is_better=False)
    )

    profile_cols = [
        "good_abs_ndi",
        "good_iqr_ndi",
        "good_rmse_log_ratio",
        "good_spearman",
        "good_n",
    ]

    hw = work[work["source"] == "HW"].set_index("alias").reindex(ordered_aliases)
    eu = work[work["source"] == "EU"].set_index("alias").reindex(ordered_aliases)

    fig_height = max(10, min(44, 0.18 * len(ordered_aliases) + 4))
    fig, axes = plt.subplots(1, 2, figsize=(18, fig_height), sharey=True)

    sns.heatmap(
        hw[profile_cols],
        ax=axes[0],
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar=True,
        yticklabels=True,
    )
    axes[0].set_title("HW metric profile")
    axes[0].set_ylabel("File alias")
    axes[0].set_xlabel("Metric goodness (higher is better)")
    axes[0].set_yticklabels(ordered_aliases, fontsize=7)
    axes[0].set_xticklabels(["|NDI|", "IQR NDI", "RMSE log", "Spearman", "N"], rotation=25)

    sns.heatmap(
        eu[profile_cols],
        ax=axes[1],
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar=True,
        yticklabels=False,
    )
    axes[1].set_title("EU metric profile")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Metric goodness (higher is better)")
    axes[1].set_xticklabels(["|NDI|", "IQR NDI", "RMSE log", "Spearman", "N"], rotation=25)

    fig.suptitle("Sensitivity Diagnostics: HW (left) vs EU (right)", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    ensure_output_dir_for_file(output_filepath)
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run compare_pop_sweep_hw_eu.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Run split HW/EU sensitivity analysis on all pop-output GPKGs."""
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="compare_pop_sweep_hw_eu", **overrides)

    data_dir = cfg["paths"]["data_dir"]
    threshold = cfg["threshold"]
    factor = cfg["eu_reference_factor"]
    eu_ref_filepath = cfg["paths"]["eu_ref_filepath"]
    max_workers = cfg["max_workers"]
    hw_weight = float(cfg["hw_weight"])

    if hw_weight < 0 or hw_weight > 1:
        raise ValueError(f"hw_weight must be within [0, 1], got {hw_weight}")

    out_dir = os.path.join(data_dir, "sensitivity", "hw_eu_pop_sweep")
    os.makedirs(out_dir, exist_ok=True)

    records = list_pop_output_files(data_dir)
    records = make_aliases(records)
    if not records:
        logger.warning("No parseable pop-output files were found. Nothing to do.")
        return

    alias_df = pd.DataFrame(records).sort_values(["alias"])
    alias_map_path = os.path.join(out_dir, "file_alias_map.csv")
    alias_df.to_csv(alias_map_path, index=False)
    logger.info("Saved alias map: %s", alias_map_path)

    summary_df = build_summary_table(
        records,
        eu_ref_filepath,
        threshold,
        factor,
        max_workers=max_workers,
    )
    if summary_df.empty:
        logger.warning("No sensitivity metrics generated from input files.")
        return

    summary_csv = os.path.join(out_dir, "hw_eu_sensitivity_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Saved summary metrics: %s", summary_csv)

    ordered_aliases, score_table = build_alias_order(summary_df, hw_weight=hw_weight)
    rank_csv = os.path.join(out_dir, "alias_rankings.csv")
    score_table.to_csv(rank_csv, index=False)
    logger.info("Saved ranking table: %s", rank_csv)

    top_candidates = score_table.head(20)
    top_path = os.path.join(out_dir, "top20_candidates.csv")
    top_candidates.to_csv(top_path, index=False)
    logger.info("Saved top candidate table: %s", top_path)

    plot_split_score_bars(
        summary_df,
        ordered_aliases,
        output_filepath=os.path.join(out_dir, "split_sensitivity_scores_hw_eu.png"),
    )
    plot_split_metric_profiles(
        summary_df,
        ordered_aliases,
        output_filepath=os.path.join(out_dir, "split_metric_profiles_hw_eu.png"),
    )

    logger.info("Sensitivity analysis complete. Outputs in %s", out_dir)


if __name__ == "__main__":
    configure_logging()
    main()
