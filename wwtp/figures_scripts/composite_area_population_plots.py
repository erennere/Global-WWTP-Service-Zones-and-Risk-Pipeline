"""Create composite histogram and scatter diagnostics for area/population ratios.

Plot A: two histograms
- total_area / zonal_sum
- round_area / total_area

Plot B: two scatter plots with 1:1 reference lines
- country aggregate vs median plant ratio for total_area / zonal_sum
- country aggregate vs median plant ratio for round_area / total_area
"""

import argparse
import os
import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..pipelines import create_pop_output_paths
from ..utils import clip_to_robust_bounds, configure_logging, ensure_output_dir_for_file, resolve_latest_zonal_sum_column


logger = logging.getLogger(__name__)


def resolve_zonal_sum_column(df, preferred):
    """Return preferred zonal-sum column or fallback to latest available year."""
    return resolve_latest_zonal_sum_column(
        df,
        preferred,
        missing_message="No '*_zonal_sum' column found in Voronoi population file.",
    )[1]


def filter_countries_by_facility_count(pop_df, boundaries, country_column, min_facility_count):
    """Filter out countries with too few facilities for stable A/B diagnostics."""
    if min_facility_count is None:
        return pop_df.copy(), boundaries.copy(), []

    min_facility_count = int(min_facility_count)
    if min_facility_count <= 0:
        return pop_df.copy(), boundaries.copy(), []

    working = pop_df.dropna(subset=[country_column]).copy()
    country_counts = working.groupby(country_column).size()
    keep_countries = country_counts[country_counts >= min_facility_count].index
    removed_countries = country_counts[country_counts < min_facility_count].sort_values(ascending=False).index.tolist()

    filtered_pop = working[working[country_column].isin(keep_countries)].copy()
    boundary_id_col = "ISO_A2" if "ISO_A2" in boundaries.columns else "ISO_A2_EH"
    filtered_boundaries = boundaries[boundaries[boundary_id_col].isin(keep_countries)].copy()
    return filtered_pop, filtered_boundaries, removed_countries


def clip_outliers(series, lower_q, upper_q, iqr_factor=1.0):
    """Drop outliers with combined quantile and IQR filtering."""
    return clip_to_robust_bounds(series, lower_q, upper_q, iqr_factor=iqr_factor)


def _trim_xy_pairs(x_vals, y_vals, lower_q=0.03, upper_q=0.97, iqr_factor=1.0):
    """Trim paired x/y values jointly so scatter ranges remain informative."""
    x = pd.to_numeric(pd.Series(x_vals), errors="coerce")
    y = pd.to_numeric(pd.Series(y_vals), errors="coerce")
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return x.iloc[0:0], y.iloc[0:0]

    x = x[mask]
    y = y[mask]

    x_trim = clip_outliers(x, lower_q=lower_q, upper_q=upper_q, iqr_factor=iqr_factor)
    y_trim = clip_outliers(y, lower_q=lower_q, upper_q=upper_q, iqr_factor=iqr_factor)
    keep_mask = x.index.isin(x_trim.index) & y.index.isin(y_trim.index)
    return x[keep_mask], y[keep_mask]


def _bleach_color(color, amount=0.35):
    """Blend a color toward white for a bleached pastel appearance."""
    rgb = np.array(color[:3], dtype=float)
    white = np.ones(3, dtype=float)
    return tuple((1.0 - amount) * rgb + amount * white)


def make_category_color_map(values):
    """Create deterministic pastel colors per category value."""
    categories = sorted(pd.Series(values).fillna("Unknown").astype(str).unique())
    base = plt.get_cmap("tab20")
    colors = {}
    for idx, cat in enumerate(categories):
        colors[cat] = _bleach_color(base(idx % 20), amount=0.45)
    return colors


def add_one_to_one_line(ax, x_vals, y_vals):
    """Add y=x dashed line spanning current value range."""
    combined = pd.concat([x_vals, y_vals], axis=0)
    combined = pd.to_numeric(combined, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if combined.empty:
        return
    lo = float(combined.min())
    hi = float(combined.max())
    if hi <= lo:
        hi = lo + 1e-6
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black", alpha=0.8)


def _set_dynamic_axis_limits(ax, values, pad_ratio=0.05):
    """Set dynamic axis limits from finite values with proportional padding."""
    clean = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return
    lo = float(clean.min())
    hi = float(clean.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(lo), 1.0) * 1e-3
    pad = span * pad_ratio
    ax.set_xlim(lo - pad, hi + pad)


def _set_dynamic_equal_xy_limits(ax, x_vals, y_vals, pad_ratio=0.05):
    """Set matching dynamic x/y limits so scatter diagnostics are comparable."""
    combined = pd.concat([pd.Series(x_vals), pd.Series(y_vals)], axis=0)
    clean = pd.to_numeric(combined, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return
    lo = float(clean.min())
    hi = float(clean.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(lo), 1.0) * 1e-3
    pad = span * pad_ratio
    low_lim = lo - pad
    high_lim = hi + pad
    ax.set_xlim(low_lim, high_lim)
    ax.set_ylim(low_lim, high_lim)


def build_country_table(pop_df, boundaries, zonal_col, color_col):
    """Create merged country table with required aggregate and median metrics."""
    d = pop_df.copy()

    d["ratio_area_pop"] = np.where(
        pd.to_numeric(d[zonal_col], errors="coerce") > 0,
        pd.to_numeric(d["total_area"], errors="coerce") / pd.to_numeric(d[zonal_col], errors="coerce"),
        np.nan,
    )
    d["ratio_round_total"] = np.where(
        pd.to_numeric(d["total_area"], errors="coerce") > 0,
        pd.to_numeric(d["round_area"], errors="coerce") / pd.to_numeric(d["total_area"], errors="coerce"),
        np.nan,
    )

    grouped = d.groupby("ISO_2", as_index=False).agg(
        total_area_sum=("total_area", "sum"),
        zonal_sum_sum=(zonal_col, "sum"),
        round_area_sum=("round_area", "sum"),
        ratio_area_pop_median=("ratio_area_pop", "median"),
        ratio_round_total_median=("ratio_round_total", "median"),
    )

    grouped["ratio_area_pop_agg"] = np.where(
        grouped["zonal_sum_sum"] > 0,
        grouped["total_area_sum"] / grouped["zonal_sum_sum"],
        np.nan,
    )
    grouped["ratio_round_total_agg"] = np.where(
        grouped["total_area_sum"] > 0,
        grouped["round_area_sum"] / grouped["total_area_sum"],
        np.nan,
    )

    boundary_id_col = "ISO_A2" if "ISO_A2" in boundaries.columns else "ISO_A2_EH"
    merged = grouped.merge(
        boundaries[[boundary_id_col, color_col]].rename(columns={boundary_id_col: "ISO_A2"}),
        left_on="ISO_2",
        right_on="ISO_A2",
        how="left",
    )
    merged[color_col] = merged[color_col].fillna("Unknown").astype(str)
    return merged


def make_histogram_plot(pop_df, zonal_col, out_path, lower_q, upper_q):
    """Generate and save Plot A with two side-by-side histograms."""
    d = pop_df.copy()
    ratio_area_pop = np.where(
        pd.to_numeric(d[zonal_col], errors="coerce") > 0,
        pd.to_numeric(d["total_area"], errors="coerce") / pd.to_numeric(d[zonal_col], errors="coerce"),
        np.nan,
    )
    ratio_round_total = np.where(
        pd.to_numeric(d["total_area"], errors="coerce") > 0,
        pd.to_numeric(d["round_area"], errors="coerce") / pd.to_numeric(d["total_area"], errors="coerce"),
        np.nan,
    )

    s_left = clip_outliers(pd.Series(ratio_area_pop), lower_q=lower_q, upper_q=upper_q, iqr_factor=1.0)
    s_right = clip_outliers(pd.Series(ratio_round_total), lower_q=lower_q, upper_q=upper_q, iqr_factor=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    axes[0].hist(s_left, bins=60, color="#90caf9", edgecolor="white", linewidth=0.6)
    axes[0].set_title("Plot A1: Distribution of Total Area / Population Proxy", fontsize=12)
    axes[0].set_xlabel(f"total_area / {zonal_col}")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.25)
    _set_dynamic_axis_limits(axes[0], s_left)

    axes[1].hist(s_right, bins=60, color="#ffcc80", edgecolor="white", linewidth=0.6)
    axes[1].set_title("Plot A2: Distribution of Circular Fraction", fontsize=12)
    axes[1].set_xlabel("round_area / total_area")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.25)
    _set_dynamic_axis_limits(axes[1], s_right)

    fig.suptitle("Plot A: Facility-Level Ratio Distributions (Outliers Trimmed)", fontsize=14, y=1.02)
    fig.tight_layout()
    ensure_output_dir_for_file(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_scatter_plot(country_df, color_col, out_path):
    """Generate and save Plot B with two side-by-side scatter plots and 1:1 lines."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), dpi=300)

    color_map = make_category_color_map(country_df[color_col])
    point_colors = country_df[color_col].map(color_map)

    x1_raw = pd.to_numeric(country_df["ratio_area_pop_agg"], errors="coerce")
    y1_raw = pd.to_numeric(country_df["ratio_area_pop_median"], errors="coerce")
    x1, y1 = _trim_xy_pairs(x1_raw, y1_raw, lower_q=0.12, upper_q=0.88, iqr_factor=0.6)
    country_df_b1 = country_df.loc[x1.index]

    axes[0].scatter(x1, y1, s=46, c=country_df_b1[color_col].map(color_map), alpha=0.9, edgecolors="white", linewidths=0.7)
    add_one_to_one_line(axes[0], x1, y1)
    axes[0].set_title("Plot B1: Aggregate vs Median (Total Area / Population Proxy)", fontsize=12)
    axes[0].set_xlabel("sum(total_area) / sum(zonal_sum)")
    axes[0].set_ylabel("median(total_area / zonal_sum)")
    axes[0].grid(True, alpha=0.3)
    _set_dynamic_equal_xy_limits(axes[0], x1, y1)

    for _, row in country_df_b1.iterrows():
        xx = pd.to_numeric(row["ratio_area_pop_agg"], errors="coerce")
        yy = pd.to_numeric(row["ratio_area_pop_median"], errors="coerce")
        if np.isfinite(xx) and np.isfinite(yy):
            axes[0].text(xx, yy, str(row["ISO_A2"]), fontsize=6.5, color=color_map[str(row[color_col])], alpha=0.9)

    x2_raw = pd.to_numeric(country_df["ratio_round_total_agg"], errors="coerce")
    y2_raw = pd.to_numeric(country_df["ratio_round_total_median"], errors="coerce")
    x2, y2 = _trim_xy_pairs(x2_raw, y2_raw, lower_q=0.03, upper_q=0.97, iqr_factor=1.0)
    country_df_b2 = country_df.loc[x2.index]

    axes[1].scatter(x2, y2, s=46, c=country_df_b2[color_col].map(color_map), alpha=0.9, edgecolors="white", linewidths=0.7)
    add_one_to_one_line(axes[1], x2, y2)
    axes[1].set_title("Plot B2: Aggregate vs Median (Circular Fraction)", fontsize=12)
    axes[1].set_xlabel("sum(round_area) / sum(total_area)")
    axes[1].set_ylabel("median(round_area / total_area)")
    axes[1].grid(True, alpha=0.3)
    _set_dynamic_equal_xy_limits(axes[1], x2, y2)

    for _, row in country_df_b2.iterrows():
        xx = pd.to_numeric(row["ratio_round_total_agg"], errors="coerce")
        yy = pd.to_numeric(row["ratio_round_total_median"], errors="coerce")
        if np.isfinite(xx) and np.isfinite(yy):
            axes[1].text(xx, yy, str(row["ISO_A2"]), fontsize=6.5, color=color_map[str(row[color_col])], alpha=0.9)

    handles = []
    labels = []
    for cat, color in color_map.items():
        handles.append(Line2D([], [], marker="o", linestyle="", color=color, markeredgecolor="white", markersize=7))
        labels.append(cat)

    if len(labels) <= 20:
        axes[1].legend(handles, labels, title=color_col, fontsize=8, title_fontsize=9, loc="best")

    fig.suptitle("Plot B: Country-Level Ratio Consistency Diagnostics", fontsize=14, y=1.02)
    fig.tight_layout()
    ensure_output_dir_for_file(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    """Parse CLI arguments for composite figure generation."""
    parser = argparse.ArgumentParser(description="Generate composite histograms and scatter plots.")
    parser.add_argument("--approach", type=str, default=None, help="Approach key for create_pop_output_paths (e.g., 0, 1, 2, 1_only_round)")
    parser.add_argument("--color-col", type=str, default="ECONOMY", help="Boundary column used for categorical color coding.")
    parser.add_argument("--zonal-col", type=str, default=None, help="Override zonal-sum column. Defaults to config zonal_sum_default_column.")
    parser.add_argument("--hist-lower-q", type=float, default=None,
                        help="Lower quantile for histogram outlier trimming (default: config plot_outlier_quantiles[0]).")
    parser.add_argument("--hist-upper-q", type=float, default=None,
                        help="Upper quantile for histogram outlier trimming (default: config plot_outlier_quantiles[1]).")

    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Entry point for composite ratio figure generation."""
    args = parse_args()
    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name="composite_area_population_plots", **overrides)

    approach = str(args.approach if args.approach is not None else cfg["figures"]["approach"])
    pop_path = os.path.abspath(create_pop_output_paths(cfg)["voronoi"][approach])
    boundaries_path = cfg["paths"]["country_boundaries_filepath"]

    pop_df = gpd.read_file(pop_path)
    boundaries = gpd.read_file(boundaries_path)

    required_cols = {"ISO_2", "total_area", "round_area"}
    missing = required_cols - set(pop_df.columns)
    if missing:
        raise KeyError(f"Missing required columns in pop filepath: {sorted(missing)}")

    zonal_col = resolve_zonal_sum_column(pop_df, args.zonal_col or cfg["zonal_sum_default_column"])
    min_country_facility_count = cfg["min_country_facility_count"]

    color_col = args.color_col
    if color_col not in boundaries.columns:
        raise KeyError(f"Color column '{color_col}' was not found in boundaries file.")

    pop_df, boundaries, removed_countries = filter_countries_by_facility_count(
        pop_df,
        boundaries,
        "ISO_2",
        min_country_facility_count,
    )
    if removed_countries:
        logger.info(
            "Filtered %s countries with fewer than %s facilities (examples: %s)",
            len(removed_countries),
            int(min_country_facility_count),
            ", ".join(removed_countries[:20]),
        )

    # Trimming quantiles come from config so this script and pop_at_risk_figures
    # clip on the same declared values; the CLI flags still override per run.
    quantiles = cfg["plot_outlier_quantiles"]
    if not isinstance(quantiles, (list, tuple)) or len(quantiles) != 2:
        raise ValueError("plot_outlier_quantiles must be a 2-item list like [0.005, 0.995]")
    lower_q = float(args.hist_lower_q if args.hist_lower_q is not None else quantiles[0])
    upper_q = float(args.hist_upper_q if args.hist_upper_q is not None else quantiles[1])

    make_histogram_plot(
        pop_df=pop_df,
        zonal_col=zonal_col,
        out_path=cfg["paths"]["composite_histogram_filepath"],
        lower_q=lower_q,
        upper_q=upper_q,
    )

    country_df = build_country_table(pop_df=pop_df, boundaries=boundaries, zonal_col=zonal_col, color_col=color_col)
    make_scatter_plot(
        country_df=country_df,
        color_col=color_col,
        out_path=cfg["paths"]["composite_scatter_filepath"],
    )


if __name__ == "__main__":
    configure_logging()
    main()
