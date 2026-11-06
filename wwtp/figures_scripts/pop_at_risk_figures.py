import argparse
import os
import logging
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.patches import Patch
from wwtp.starter import add_standard_override_arguments, load_config, parse_config_overrides
from wwtp.utils import configure_logging, robust_bounds
from wwtp.pop_at_risk_river_calculations.find_pop_in_danger_pop import find_tiles_in_countries

logger = logging.getLogger(__name__)


def _robust_bounds(values, positive_only=False, quantile_range=(0.02, 0.98), iqr_factor=1.5):
    """Estimate robust plotting bounds by combining quantile and IQR filtering."""
    return robust_bounds(
        values,
        quantile_range=quantile_range,
        iqr_factor=iqr_factor,
        positive_only=positive_only,
    )


def create_single_plot(
    z8_stats,
    column,
    title,
    output_filename,
    output_dir="coverage_maps",
    cmap="viridis",
    scale_type="log",
    value_transform=None,
    value_label=None,
    min_count_col=None,
    min_count=None,
    projection=None,
    vmin=None,
    vmax=None,
    missing_color="gray",
    edgecolor="white",
    linewidth=0.5,
    legend=True,
    legend_kwds=None,
    suptitle_template="Overview of the {title}",
    author_note=None,
    save_dpi=1000,
    show=True,
    outlier_quantiles=(0.02, 0.98),
    outlier_iqr_factor=1.5,
    country_id_col=None,
    nodata_country_color=None,
    nodata_country_label="NODATA",
    min_display_value=None,
):
    """Create a projected choropleth plot from a GeoDataFrame column.

    Parameters are intentionally generic so this helper can be reused for
    different geospatial completeness or risk metrics.
    """
    cmap = plt.get_cmap(cmap)  # a name or a Colormap
    if column not in z8_stats.columns:
        raise KeyError(f"Column '{column}' not found in GeoDataFrame")

    gdf = z8_stats.copy()

    # Set CRS to WGS84 if missing.
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)

    projection = projection or ccrs.Robinson()
    fig, ax = plt.subplots(figsize=(20, 12), dpi=400, subplot_kw={"projection": projection})

    # Build plotting values from source column.
    values = pd.to_numeric(gdf[column], errors="coerce")
    if value_transform is not None:
        values = value_transform(values)
    gdf["plot_value"] = values

    # Remove invalid geometries and project.
    gdf = gdf[gdf["geometry"].is_valid]
    gdf = gdf.to_crs(projection.proj4_init)

    # Optional masking by supporting count column.
    if min_count_col is not None and min_count is not None:
        if min_count_col not in gdf.columns:
            raise KeyError(f"Column '{min_count_col}' not found in GeoDataFrame")
        gdf.loc[gdf[min_count_col] < min_count, "plot_value"] = np.nan

    # Apply a single explicit floor cutoff for display, independent of normalization bounds.
    if min_display_value is not None:
        gdf.loc[gdf["plot_value"] < float(min_display_value), "plot_value"] = np.nan

    if scale_type == "log":
        positive_values = gdf["plot_value"][gdf["plot_value"] > 0]
        if min_display_value is not None:
            valid_values = positive_values[positive_values >= float(min_display_value)]
        else:
            valid_values = positive_values

        if valid_values.empty:
            raise ValueError("No valid positive values in 'plot_value' for logarithmic normalization.")

        auto_vmin, auto_vmax = _robust_bounds(
            valid_values,
            positive_only=True,
            quantile_range=outlier_quantiles,
            iqr_factor=outlier_iqr_factor,
        )

        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax
        if min_display_value is not None:
            vmin = max(float(vmin), float(min_display_value))
            if vmax <= vmin:
                vmax = vmin * 10.0
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    elif scale_type == "linear":
        linear_values = gdf["plot_value"]
        if min_display_value is not None:
            linear_values = linear_values[linear_values >= float(min_display_value)]

        auto_vmin, auto_vmax = _robust_bounds(
            linear_values,
            positive_only=False,
            quantile_range=outlier_quantiles,
            iqr_factor=outlier_iqr_factor,
        )
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax
        if min_display_value is not None:
            vmin = max(float(vmin), float(min_display_value))
        if vmin == vmax:
            raise ValueError("Minimum and maximum values are the same. Cannot normalize.")
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    else:
        raise ValueError("scale_type must be 'log' or 'linear'")

    cmap_values = cmap(np.arange(cmap.N))
    cmap = ListedColormap(cmap_values)
    cmap.set_bad(color=missing_color)

    default_legend_kwds = {
        "label": value_label or title,
        "anchor": (0.2, 0.75),
        "orientation": "horizontal",
        "shrink": 0.25,
        "pad": 0.05,
    }
    if legend_kwds:
        default_legend_kwds.update(legend_kwds)

    nodata_patch = None
    gdf_plot = gdf
    if country_id_col is not None and country_id_col in gdf.columns and nodata_country_color is not None:
        has_value_by_country = gdf.groupby(country_id_col)["plot_value"].apply(lambda s: s.notna().any())
        nodata_mask = ~gdf[country_id_col].map(has_value_by_country).fillna(False)
        gdf_nodata = gdf[nodata_mask]
        gdf_plot = gdf[~nodata_mask]

        if not gdf_nodata.empty:
            gdf_nodata.plot(
                ax=ax,
                edgecolor=edgecolor,
                linewidth=linewidth,
                color=nodata_country_color,
            )
            nodata_patch = Patch(facecolor=nodata_country_color, edgecolor=edgecolor, label=nodata_country_label)

    gdf_plot.plot(
        ax=ax,
        edgecolor=edgecolor,
        linewidth=linewidth,
        column="plot_value",
        cmap=cmap,
        legend=legend,
        legend_kwds=default_legend_kwds,
        norm=norm,
        missing_kwds={"color": missing_color},
    )

    if nodata_patch is not None:
        existing_legend = ax.get_legend()
        if existing_legend is not None:
            ax.add_artist(existing_legend)
        ax.legend(handles=[nodata_patch], loc="lower left", frameon=True, fontsize=9)

    if hasattr(ax, "gridlines"):
        gl = ax.gridlines(linewidth=0.5, color="gray", alpha=0.5, linestyle="--", draw_labels=True)  # type: ignore[attr-defined]
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

    plt.suptitle(suptitle_template.format(title=title), fontsize=16, fontweight="bold", y=0.95)

    if author_note:
        ax.annotate(
            author_note,
            xy=(1, -0.2),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=8,
            backgroundcolor="white",
        )

    plt.tight_layout(rect=(0, 0, 1, 0.9), pad=2)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(fname=f"{output_dir}/{output_filename}", dpi=save_dpi)
    if show:
        plt.show()
    return fig, ax


def create_impact_polygon_plots(
    pop_at_risk_gdf,
    tiles_gdf,
    output_filepath,
    save_dpi=1000,
    outlier_quantiles=(0.005, 0.995),
    outlier_iqr_factor=3.0,
    country_id_col=None,
    nodata_country_color=None,
    nodata_country_label="NODATA",
    min_display_value=None,
):
    """Create one map per radius/year population zonal-sum column."""
    tiles_gdf = tiles_gdf.copy()
    pop_at_risk_gdf = pop_at_risk_gdf.copy()
    tiles_gdf = pd.merge(tiles_gdf, pop_at_risk_gdf.drop(columns="geometry"), on="tile", how="left")
    plot_columns = [col for col in tiles_gdf.columns if col.endswith("_zonal_sum")]

    if not plot_columns:
        logger.warning("No '*_zonal_sum' columns found for impact polygon plotting.")
        return

    colormap_cycle = [
        plt.get_cmap("viridis"),
        plt.get_cmap("plasma"),
        plt.get_cmap("cividis"),
        plt.get_cmap("magma"),
        plt.get_cmap("YlGnBu"),
        plt.get_cmap("YlOrRd"),
        plt.get_cmap("PuBuGn"),
        plt.get_cmap("cubehelix"),
    ]

    for idx, col in enumerate(plot_columns):
        # Every '*_zonal_sum' name splits into at least 3 parts, so only the
        # numeric parse can fail.
        parts = col.split("_")
        try:
            radius = int(parts[0])
            year = int(parts[1])
        except ValueError:
            logger.warning("Skipping zonal-sum column with non-numeric radius/year: %s", col)
            continue

        title = f"Population at Risk (Radius {radius}m, Year {year})"
        output_filename = f"impact_pop_radius_{radius}_year_{year}.png"
        selected_cmap = colormap_cycle[idx % len(colormap_cycle)]

        create_single_plot(
            z8_stats=tiles_gdf,
            column=col,
            title=title,
            output_filename=output_filename,
            output_dir=output_filepath,
            cmap=selected_cmap,
            scale_type="log",
            value_label="Population at risk",
            suptitle_template="{title}",
            save_dpi=save_dpi,
            show=False,
            outlier_quantiles=outlier_quantiles,
            outlier_iqr_factor=outlier_iqr_factor,
            country_id_col=country_id_col,
            nodata_country_color=nodata_country_color,
            nodata_country_label=nodata_country_label,
            min_display_value=min_display_value,
        )
        logger.info("Saved impact polygon map for column: %s (cmap=%s)", col, selected_cmap.name)


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run pop_at_risk_figures.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    overrides = parse_config_overrides(args=parse_args())

    cfg = load_config(script_name="pop_at_risk_figures", **overrides)
    country_boundaries_gdf = gpd.read_file(cfg["paths"]["country_boundaries_filepath"])
    country_id_col = cfg["country_id_column"]
    if country_id_col not in country_boundaries_gdf.columns:
        raise KeyError(
            f"Configured country_id_column '{country_id_col}' not found in boundaries; "
            f"available: {sorted(country_boundaries_gdf.columns)}"
        )

    quantiles = cfg["plot_outlier_quantiles"]
    if not isinstance(quantiles, (list, tuple)) or len(quantiles) != 2:
        raise ValueError("plot_outlier_quantiles must be a 2-item list like [0.005, 0.995]")
    outlier_quantiles = (float(quantiles[0]), float(quantiles[1]))
    outlier_iqr_factor = float(cfg["plot_outlier_iqr_factor"])

    max_workers = int(cfg["max_workers"])
    zoom_level = int(cfg["zoom_level"])
    save_dpi = int(cfg["save_dpi"])
    nodata_country_color = cfg["nodata_country_color"]
    nodata_country_label = cfg["nodata_country_label"]
    min_display_value = float(cfg["min_display_population"])

    pop_at_risk_path = cfg["paths"]["pop_at_risk_output_filepath"]
    pop_at_risk_gdf = gpd.read_parquet(pop_at_risk_path)
    tiles_gdf = find_tiles_in_countries(
        country_boundaries_gdf, zoom_level=zoom_level, country_id_col=country_id_col, max_workers=max_workers
    )
    create_impact_polygon_plots(
        pop_at_risk_gdf,
        tiles_gdf,
        cfg["paths"]["figures_dir"],
        save_dpi,
        outlier_quantiles=outlier_quantiles,
        outlier_iqr_factor=outlier_iqr_factor,
        country_id_col=country_id_col,
        nodata_country_color=nodata_country_color,
        nodata_country_label=nodata_country_label,
        min_display_value=min_display_value,
    )

    unserved_df_filepath = cfg["paths"]["non_served_above_threshold_outpath"].replace(".csv", ".gpkg")
    unserved_df = gpd.read_file(unserved_df_filepath, columns=["tile", "pop_sum"])[["tile", "pop_sum"]]
    unserved_df["pop_sum"] = pd.to_numeric(unserved_df["pop_sum"], errors="coerce")
    unserved_df = unserved_df.groupby("tile", as_index=False)["pop_sum"].sum()
    tiles_gdf = pd.merge(tiles_gdf, unserved_df, on="tile", how="left")
    create_single_plot(
        z8_stats=tiles_gdf,
        column="pop_sum",
        title="Unserved Population",
        output_filename="unserved_population_tiles.png",
        output_dir=cfg["paths"]["figures_dir"],
        cmap=plt.get_cmap("inferno"),
        scale_type="log",
        value_label="Unserved population",
        suptitle_template="{title}",
        save_dpi=save_dpi,
        show=False,
        outlier_quantiles=outlier_quantiles,
        outlier_iqr_factor=outlier_iqr_factor,
        country_id_col=country_id_col,
        nodata_country_color=nodata_country_color,
        nodata_country_label=nodata_country_label,
        min_display_value=min_display_value,
    )
    logger.info("Saved unserved population tile map.")


if __name__ == "__main__":
    configure_logging()
    main()
