"""Compare project population estimates against HydroWaste reference values.

The script builds yearly comparison metrics for verification subsets and saves
histogram panels for both normalized-difference and multiplicative views.
"""

import argparse
import os
import logging
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from .. import provenance
from ..utils import configure_logging, ensure_output_dir_for_file, reported_population_mask

logger = logging.getLogger(__name__)


def ndvi(df, col1, col2, new_col, small_value=0.001):
    """Compute a normalized difference index between two numeric columns."""
    df = df.copy()
    df[new_col] = (df[col1] - df[col2]) / (df[col1] + df[col2] + small_value)
    return df


def extract_voronoi_parameters(filepath):
    """Extract all parameters from filepath: version, level, buffer, weight_type, approach, only_round, weight_func.

    Expected filepath structure:
    {data_dir}/voronoi_layers/v{version}/lvl{level}/bf{buffer}/{weight_type}/appr_{approach}_v{version}_lvl{level}_bf{buffer}{weight_func}.gpkg
    """
    params = {}

    # Normalize path separators for consistent parsing
    normalized_path = filepath.replace("\\", "/")
    path_parts = normalized_path.split("/")

    # Parse path components (voronoi_layers/v{version}/lvl{level}/bf{buffer}/{weight_type})
    for i, part in enumerate(path_parts):
        if part.startswith("v") and len(part) > 1 and part[1].isdigit():
            params["version"] = part[1:]  # e.g., 'v2' -> '2'
        elif part.startswith("lvl"):
            params["level"] = int(part[3:])  # e.g., 'lvl3' -> 3
        elif part.startswith("bf"):
            match = re.match(r"bf(\d+)", part)
            if match:
                params["buffer"] = int(match.group(1))
        elif (
            part
            and not part.startswith("appr")
            and part not in ("voronoi_layers", "voronoi")
            and not part.endswith(".gpkg")
        ):
            # This is likely the weight_type directory (e.g., 'linear', 'logarithmic')
            if i > 0 and path_parts[i - 1].startswith("bf"):
                params["weight_type"] = part

    # Parse filename components
    filename = os.path.basename(filepath)

    # Extract approach number and only_round flag
    match = re.search(r"appr_(\d+)(?:_only_round)?", filename)
    if match:
        params["approach"] = match.group(1)
        params["only_round"] = "_only_round" in filename

    # Extract weight_func (everything between bf{buffer} and .gpkg)
    match = re.search(r"bf\d+(.+?)\.gpkg", filename)
    if match:
        params["weight_func"] = match.group(1)

    return params


def multiples(df, col1, col2, new_col, small_value=0.001):
    """Compute the multiplicative comparison ratio between two columns."""
    df = df.copy()
    df[new_col] = (df[col1] - df[col2]) / (df[col2] + small_value) + 1
    return df


def replace_inf(df, col):
    """Replace infinite values in one column with NaN for plotting safety."""
    df = df.copy()
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def composite_histogram(
    data,
    my_dict,
    title,
    output_filepath=None,
    save=False,
    dpi=300,
    ylabel="N_WWTPs",
    xlabel=None,
    bins=100,
    lower_quantile=0.01,
    upper_quantile=0.95,
    fontsize=26,
    small_fontsize=18,
    hide_empty_axis=True,
):
    """Plot a grid of histograms for multiple year-specific comparison columns.

    Quantile clipping is applied per column before plotting so extreme outliers
    do not dominate the axis range.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    pastel_colors = sns.color_palette("pastel", n_colors=len(my_dict))

    for i, (year, col_name) in enumerate(my_dict.items()):
        row, col = divmod(i, 5)
        ax = axes[row, col]

        # CHECK IF COLUMN EXISTS AND HAS DATA
        if col_name not in data.columns or data[col_name].dropna().empty:
            logger.warning(f"Column '{col_name}' is missing or empty in the data. Skipping plot for year {year}.")
            ax.set_title(f"{year} (No Data)")
            if hide_empty_axis:
                ax.axis("off")  # Hide empty plots
            continue

        color = pastel_colors[i]

        vmin = data[col_name].quantile(lower_quantile)
        vmax = data[col_name].quantile(upper_quantile)

        # CHECK IF QUANTILES ARE FINITE
        if np.isnan(vmin) or np.isnan(vmax):
            logger.warning(f"Column '{col_name}' has invalid quantile range for year {year}. Skipping plot.")
            ax.set_title(f"{year} (Invalid Range)")
            continue

        subset = data[(data[col_name] >= vmin) & (data[col_name] <= vmax)][col_name]
        if subset.empty:
            logger.warning(f"Column '{col_name}' has no data within quantile range for year {year}. Skipping plot.")
            continue

        # Plot histogram
        ax.hist(subset, bins=bins, range=(vmin, vmax), color=color, edgecolor="black")

        mean_val = subset.mean()
        median_val = subset.median()
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="gray", linestyle="--", linewidth=1.5, label=f"Median: {median_val:.2f}")

        N = len(subset)
        ax.set_title(f"{year}, N : {N}", fontsize=small_fontsize)

        if col != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(ylabel, fontsize=small_fontsize)
        if row == 1:
            ax.set_xlabel(xlabel if xlabel else "", fontsize=small_fontsize)

        ax.grid(True)
        ax.legend(fontsize=8, loc="upper right", frameon=False)

    fig.suptitle(title, fontsize=fontsize)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save and output_filepath:
        ensure_output_dir_for_file(output_filepath)
        plt.savefig(output_filepath, dpi=dpi)

    plt.show()
    plt.close(fig)


def orchestrate_single(
    gdf,
    approach,
    plot_args,
    output_dir,
    filename,
    pop_col="POP_SERVED",
    qual_pop_default=None,
    filter_qual_pop=True,
    upper_quantile_hw_comp=1,
    comp_output_prefix="hw_comp",
    reference_name="HydroWaste",
    hide_empty_axis=True,
):
    """Compute reference-comparison metrics for one verification file.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Verification subset to analyse.
    approach : str | None
        Approach identifier parsed from the file name.
    plot_args : dict
        Keyword arguments forwarded to ``composite_histogram``.
    output_dir : str
        Directory where plot images are written.
    filename : str
        Source verification filename.
    pop_col : str, default='POP_SERVED'
        Population reference column used for comparisons.
    qual_pop_default : str | None, default=None
        Value used to backfill a missing ``QUAL_POP`` column. The default
        ``None`` does not backfill, so a missing column raises when
        ``filter_qual_pop`` is set instead of marking every plant as reported.
    filter_qual_pop : bool, default=True
        Whether to keep only rows with reported population (``QUAL_POP == 1``).
    upper_quantile_hw_comp : float, default=1
        Upper quantile used to clip the multiplicative-comparison histogram.
    comp_output_prefix : str, default='hw_comp'
        Filename prefix for the multiplicative-comparison plot.
    reference_name : str, default='HydroWaste'
        Reference dataset name used in plot titles.
    hide_empty_axis : bool, default=True
        Whether ``composite_histogram`` should hide empty subplot axes.
    """
    if pop_col not in gdf.columns:
        raise KeyError(f"Missing required population column '{pop_col}'")

    years_and_cols = dict(
        sorted({int(col.split("_")[0]): col for col in gdf.columns if col.endswith("_zonal_sum")}.items())
    )
    ndi_dict = {}
    HW_comp_dict = {}
    gdf["indx"] = range(len(gdf))
    if qual_pop_default is not None and "QUAL_POP" not in gdf.columns:
        gdf["QUAL_POP"] = qual_pop_default
    reported = reported_population_mask(gdf).to_numpy() if filter_qual_pop else None

    verified = "single"
    if "unver" in filename:
        verified = False
    elif "ver" in filename:
        verified = True

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for year, col in years_and_cols.items():
        if year == 2014:
            continue
        ndi_col = f"{year}_NDI"
        HW_comp_col = f"{year}_HW_comp"
        ndi_dict[year] = ndi_col
        HW_comp_dict[year] = HW_comp_col

        if filter_qual_pop:
            pop_file = gdf[(gdf[pop_col].notna()) & (gdf[col].notna()) & reported]
        else:
            pop_file = gdf[(gdf[pop_col].notna()) & (gdf[col].notna())]

        pop_file = ndvi(pop_file, col, pop_col, ndi_col)
        pop_file = multiples(pop_file, col, pop_col, HW_comp_col)
        pop_file = replace_inf(pop_file, ndi_col)
        pop_file = replace_inf(pop_file, HW_comp_col)
        pop_file = pop_file[["indx", ndi_col, HW_comp_col]]
        gdf = pd.merge(gdf, pop_file, on="indx", how="left")

    ylabel = "N_WWTPs"
    xlabel_ndi = "NDI"
    xlabel_hW_comp = r"$\alpha$"
    upper_quantile_ndi = 0.99
    ndi_output_filepath = os.path.join(output_dir, f"ndi_{filename.replace('.gpkg', '.png')}")
    hw_comp_output_filepath = os.path.join(output_dir, f"{comp_output_prefix}_{filename.replace('.gpkg', '.png')}")

    ndi_title = f"Normalized Difference Index (NDI) w.r.t. {reference_name}, approach: {approach}\n ver: {verified}"
    hw_comp_title = rf"Population = $\alpha\cdot${reference_name}, approach: {approach}" + f"\n ver: {verified}"
    composite_histogram(
        gdf,
        ndi_dict,
        ndi_title,
        output_filepath=ndi_output_filepath,
        ylabel=ylabel,
        xlabel=xlabel_ndi,
        upper_quantile=upper_quantile_ndi,
        hide_empty_axis=hide_empty_axis,
        **plot_args,
    )
    composite_histogram(
        gdf,
        HW_comp_dict,
        hw_comp_title,
        output_filepath=hw_comp_output_filepath,
        ylabel=ylabel,
        xlabel=xlabel_hW_comp,
        upper_quantile=upper_quantile_hw_comp,
        hide_empty_axis=hide_empty_axis,
        **plot_args,
    )


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run hw_comparison.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Load verification files and export HydroWaste comparison histograms.

    Returns
    -------
    None
        The function iterates over configured verification files and writes the
        generated comparison plots to disk.
    """
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="hw_comparison", **overrides)
    provenance.write_run_manifest(cfg, "hw_comparison")
    ver_dir = cfg["paths"]["verification_dir"]
    plots_dir = cfg["paths"]["hw_plots_dir"]
    if not os.path.isdir(ver_dir):
        logger.warning("Verification directory not found: %s", ver_dir)
        return

    pop_filepaths = [os.path.join(ver_dir, f) for f in os.listdir(ver_dir) if f.endswith(".gpkg")]
    plot_args = {"dpi": 300, "bins": 100, "save": True, "fontsize": 26, "small_fontsize": 18, "lower_quantile": 0.01}
    pop_col = "POP_SERVED"
    for filepath in pop_filepaths:
        filename = os.path.basename(filepath)
        params = extract_voronoi_parameters(filepath)
        approach = params.get("approach") if isinstance(params, dict) else None
        gdf = gpd.read_file(filepath)
        orchestrate_single(gdf, approach, plot_args, plots_dir, filename, pop_col)


if __name__ == "__main__":
    configure_logging()
    main()
