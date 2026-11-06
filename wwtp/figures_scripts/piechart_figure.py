"""Create a static world map summarizing served population and WWTP mix.

The figure combines a choropleth background with country-level donut markers
that split residential and industrial WWTP area indicators.
"""

import argparse
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Circle, Patch
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..pipelines import create_pop_output_paths
from ..utils import configure_logging, ensure_output_dir_for_file, industrial_category_mask, resolve_latest_zonal_sum_column
from ._shared import FULL_STATS, ensure_population_percentage_column
from . import _shared

def aggregate_by_country(gdf, country_column, agg_column, industrial_column=None, is_pop=False):
    """Aggregate facility-level attributes to country-level summary statistics."""
    return _shared.aggregate_by_country(
        gdf, country_column, agg_column,
        industrial_column=industrial_column, is_pop=is_pop, stats=FULL_STATS,
    )

def plot_splitted_piechart(dist_tag1, dist_tag2, ax,
                            size_tag1, size_tag2, min_size,
                            labels=False, labels_text = ['Paved', 'Unpaved', ''],
                            cmap="tab20c"):
    """Draw a two-sided donut pie chart comparing residential and industrial shares.

    `dist_tag1` and `dist_tag2` hold the category counts for the left and right
    halves, while `size_tag1` and `size_tag2` control the relative radii of the
    two pies.
    """
    ax.grid(False)
    ax.set_axis_off()
    
    # Determine relative radii
    rad_tag1, rad_tag2 = (1, size_tag2/size_tag1) if size_tag1 > size_tag2 else (size_tag1/size_tag2, 1)
    # Colormap
    cmap = plt.get_cmap(cmap)
    colors = {
        "tag1": [cmap(i) for i in [2, 6]] + [(1,1,1,0)],
        "tag2": [cmap(i) for i in [1, 5]] + [(1,1,1,0)]
    }
    
    # Append totals for donut effect
    val_tag1 = np.array(dist_tag1 + [sum(dist_tag1)])
    val_tag2 = np.array(dist_tag2 + [sum(dist_tag2)])
    pie_labels =  labels_text if labels else None

    def plot_pie(values, radius, cols, min_size, startangle, counterclock):
        """Draw one half-donut only when the requested marker is large enough."""
        # Skip pies that would render below the configured minimum visible size.
        if radius <= 0 or sum(values)/2 < min_size:
            return
        wedgeprops = dict(width=0.7) if radius > 0.5 else None
        wedges, _ = ax.pie(values, radius=radius, colors=cols,
                            wedgeprops=wedgeprops, labels=pie_labels,
                            startangle=startangle, counterclock=counterclock,
                            textprops={"fontsize":12})
        for w in wedges[:-1]:
            w.set_edgecolor('white')
            w.set_linewidth(0.9)
    plot_pie(val_tag1, rad_tag1, colors["tag1"], min_size, startangle=90, counterclock=True)
    plot_pie(val_tag2, rad_tag2, colors["tag2"], min_size, startangle=90, counterclock=False)

    ax.set_axis_off()

def get_pos(geometry):
    """Return a representative plotting position for polygon or multipolygon geometry."""
    if geometry.geom_type == 'Polygon':
        return geometry.centroid.x, geometry.centroid.y
    elif geometry.geom_type == 'MultiPolygon':
        return max(list(geometry.geoms), key=lambda x: x.area).centroid.x, (max(list(geometry.geoms), key=lambda x: x.area)).centroid.y
    raise ValueError("Invalid geometry type")

def calculate_size(value, min_value, max_value, min_size, max_size, scale='log'):
    """Map a value to a plotted pie size using log or linear scaling."""
    return _shared.calculate_size(
        value, min_value, max_value, min_size, max_size,
        scale=scale, degenerate='mid', floor_nonpositive=False,
    )


def round_numbers(arr, breaks):
    """Generate rounded legend break labels spanning the observed value range."""
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return breaks
    nums = np.linspace(arr.min(), arr.max(), len(breaks)).astype(int)
    rounded = []
    for n in nums:
        log_n  = np.log10(n)
        power = int(np.floor(log_n))
        coeff = round(10*(log_n - power))
        rounded.append((coeff+1)*10**power)
    return rounded


def resolve_zonal_sum_columns(df, preferred):
    """Resolve preferred zonal-sum column with fallback to latest available year."""
    return resolve_latest_zonal_sum_column(
        df,
        preferred,
        missing_message="No '*_zonal_sum' column found in population layer.",
    )[1]
            
def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run piechart_figure.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Create the static global WWTP-type summary figure and save it to disk."""
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="piechart_figure", **overrides)

    approach = str(cfg['figures']['approach'])
    boundaries_filepath = cfg['paths']['country_boundaries_filepath']
    output_paths = create_pop_output_paths(cfg)
    voronoi_map = output_paths['voronoi']
    pop_filepath = os.path.abspath(voronoi_map[approach])
    stats_filepath = cfg['paths']['raster_country_stats_filepath']

    pop_column = 'population_served_index'
    zonal_sum_col = cfg['zonal_sum_default_column']
    filter_col = zonal_sum_col
    industrial_col = 'IND/RES'
    tag1 = 'round_area'
    tag2 = 'wwtp_area_rect_2'
    min_total_size = float(cfg['min_total_size'])
    nodata_country_color = cfg['nodata_country_color']
    nodata_country_label = cfg['nodata_country_label']
    scale = 'linear'
    agg_type = 'sum'

    agg_columns = {
        True: [zonal_sum_col],
        False: ['num_detection_circle', 'num_detection_rect', 'total_area', tag1, tag2]}
    
    # Load boundaries
    boundaries = gpd.read_file(boundaries_filepath).to_crs("ESRI:54030")
    boundaries['country'] = boundaries['ISO_A2_EH']
    boundaries = boundaries.drop_duplicates(subset=['country'])

    # Load population / WWTP data
    pop_gdf = gpd.read_file(pop_filepath)
    pop_gdf['country'] = pop_gdf['ISO_2']
    pop_gdf = pop_gdf.drop('geometry', axis=1)
    zonal_sum_col = resolve_zonal_sum_columns(pop_gdf, zonal_sum_col)
    filter_col = zonal_sum_col
    agg_columns[True] = [zonal_sum_col]
    # Mixed-use counts as industrial here too, so this split matches the
    # unconnected-industrial layer instead of disagreeing with it.
    industrial_mask = industrial_category_mask(
        pop_gdf, cfg['industrial_category_numbers'], cfg['mix_use_categories']
    )
    if industrial_mask is None:
        raise KeyError(
            "Column 'category_number' is required to split industrial from "
            f"residential sites; columns present: {sorted(pop_gdf.columns)}"
        )
    pop_gdf[industrial_col] = industrial_mask

    if not os.path.exists(stats_filepath):
        raise FileNotFoundError(f"Stats file not found: {stats_filepath}")
    stats_df = pd.read_csv(stats_filepath)
    stats_df.columns = [str(c).strip().lstrip('\ufeff') for c in stats_df.columns]
    
    agg_datasets = []
    for is_pop, col_list in agg_columns.items():
        for agg_column in col_list:
            agg_datasets.append(aggregate_by_country(pop_gdf, 'country', agg_column, industrial_col, is_pop=is_pop))
    for dataset in agg_datasets:
        boundaries = boundaries.merge(dataset, on='country', how='left')
    boundaries = boundaries.merge(stats_df, on='country', how='left')

    # Disable seaborn/default style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 10), dpi=600)
    ax = fig.add_axes((0.05, 0.15, 0.9, 0.8), projection=ccrs.Robinson())  # [left, bottom, width, height]
        
    # Plot boundaries colored by population / WWTP metric
    boundaries = boundaries.drop_duplicates(subset=['country'])
    #boundaries[pop_column] = boundaries[pop_column]/1000
    pop_column = ensure_population_percentage_column(boundaries, pop_column, zonal_sum_col)
    boundaries[pop_column] = pd.to_numeric(boundaries[pop_column], errors='coerce') * 100
    """     boundaries[boundaries.geometry.notna()].plot(
        ax=ax,
        column=pop_column,
        cmap='viridis',
        edgecolor='white',
        linewidth=0.5,
        legend=True,
        norm=LogNorm(),  # <-- THIS
        legend_kwds={
            'label': "Number of People Served by a WWTP (in thousands)",
            'orientation': 'horizontal',
            'shrink': 0.25,
            'pad': 0.05,
            'aspect' : 20
        }
    ) """
    # Make a copy of your GeoDataFrame with valid geometries
    gdf = boundaries[boundaries.geometry.notna()].copy()
    gdf_nodata = gdf[gdf[pop_column].isna()].copy()
    gdf_plot = gdf[gdf[pop_column].notna()].copy()

    data = gdf_plot[pop_column].dropna()
    if data.empty:
        log_norm = LogNorm(vmin=1.0, vmax=10.0)
        norm = Normalize(vmin=0.0, vmax=1.0)
    else:
        log_norm = LogNorm(vmin=data.min(), vmax=data.max())
        norm = Normalize(vmin=data.min(), vmax=data.max())
    
    # Plot choropleth WITHOUT automatic legend
    if not gdf_nodata.empty:
        gdf_nodata.plot(
            ax=ax,
            color=nodata_country_color,
            edgecolor='white',
            linewidth=0.5,
            legend=False,
        )

    if not gdf_plot.empty:
        gdf_plot.plot(
            ax=ax,
            column=pop_column,
            cmap='viridis',
            edgecolor='white',
            linewidth=0.5,
            legend=False,
        )
    ax.set_global()

    # ScalarMappable for manual colorbar
    sm = ScalarMappable(cmap='viridis', norm=log_norm)
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm._A = []

    # Colorbar below the map
    cbar_ax = fig.add_axes((0.3, 0.1, 0.5, 0.02))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

    #cbar.set_label("Number of People Served by a WWTP (in thousands)", fontsize=20)
    cbar.set_label("Percentage of People Served by a WWTP", fontsize=20)

    # Add coastlines for context
    ax.coastlines(resolution='110m', color='black', linewidth=0.5)
    if not gdf_nodata.empty:
        nodata_patch = Patch(facecolor=nodata_country_color, edgecolor='white', label=nodata_country_label)
        ax.legend(handles=[nodata_patch], loc='upper left', frameon=True, fontsize=9)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    size_cols = [
        f'IND_{tag1}_{agg_type}',
        f'IND_{tag2}_{agg_type}',
        f'RES_{tag1}_{agg_type}',
        f'RES_{tag2}_{agg_type}',
    ]
    size_df = boundaries[size_cols].apply(pd.to_numeric, errors='coerce')
    boundaries['total_size'] = size_df.T.sum(min_count=1).fillna(0)

    valid_total_size = boundaries['total_size'][np.isfinite(boundaries['total_size']) & (boundaries['total_size'] > 0)]
    if valid_total_size.empty:
        min_size = 1.0
        max_size = 1.0
    else:
        max_size = float(np.nanpercentile(valid_total_size, 99))
        min_size = float(np.nanpercentile(valid_total_size, 1))
    min_pie_size = 0.1
    max_pie_size = 0.7
    breaks = [5, 15, 30, 45, 60, 75, 90]
    breaks = round_numbers(boundaries['total_size'], breaks)
    
    for index, row in boundaries.iterrows():
        xpos, ypos = get_pos(row.geometry)
        dist_ind = [float(row[f'IND_{tag1}_{agg_type}']), float(row[f'IND_{tag2}_{agg_type}'])]
        dist_res = [float(row[f'RES_{tag1}_{agg_type}']), float(row[f'RES_{tag2}_{agg_type}'])]
        """         
        if dist_ind == [0., 0.] or any(np.isnan(item) for item in dist_ind):
            dist_ind = [1., 1.]
            
        if dist_res == [0., 0.] or any(np.isnan(item) for item in dist_res):
            dist_res = [1., 1.] 
        """
        if any(np.isnan(item) for item in dist_ind):
            dist_ind = [0., 0.]
        if any(np.isnan(item) for item in dist_res):
            dist_res = [0., 0.]
        if max(sum(dist_ind), sum(dist_res)) < min_total_size:
            continue

        #size_ind = max(calculate_size(sum(dist_ind), min_size, max_size, min_pie_size, max_pie_size, scale), min_pie_size)
        size_ind = calculate_size(sum(dist_ind), min_size, max_size, min_pie_size, max_pie_size, scale)
        size_res = calculate_size(sum(dist_res), min_size, max_size, min_pie_size, max_pie_size, scale)

        size = size_ind + size_res
        if not np.isfinite(size) or size <= 0:
            continue
        if size_res < min_pie_size and size_ind < min_pie_size:
            continue
        ax_pie = inset_axes(ax, width=size, height=size, loc='center', bbox_to_anchor=(xpos, ypos, 1, 1), bbox_transform=ax.transData)
        plot_splitted_piechart(dist_res, dist_ind, ax_pie, size_res, size_ind, min_pie_size)

    # Create size legend directly in the figure
    largest_size = calculate_size(breaks[-1], min_size, max_size, min_pie_size, max_pie_size, scale)
    if not np.isfinite(largest_size) or largest_size <= 0:
        largest_size = max_pie_size
    legend_ax = inset_axes(ax, width=largest_size, height=largest_size, loc='lower left', bbox_to_anchor=(0.06, 0.02, 1, 1), bbox_transform=ax.transAxes)
    legend_ax.axis('off')

    # Keep the classic stacked-circle legend, but place it a bit higher and with a wider value range.
    legend_sizes = [5000000, 10000000, 20000000, 40000000, 80000000, 120000000]
    y_base = 0.34
    for size in legend_sizes:
        relative_size = calculate_size(size, min_size, max_size, min_pie_size, max_pie_size, scale) / largest_size
        if not np.isfinite(relative_size) or relative_size <= 0:
            continue
        circle = Circle((0.5, y_base + relative_size / 2), relative_size / 2, color='black', fill=False)
        legend_ax.add_patch(circle)
        legend_ax.annotate(
            str(round(size / 10**6)) + r" $\text{km}^2$",
            xy=(0.5, y_base + relative_size),
            xytext=(1.05, y_base + relative_size),
            ha='left',
            va='center',
            arrowprops=dict(arrowstyle='-', color='black'),
            fontsize=8,
        )
    legend_ax.set_title("Total WWTP Area", fontsize=14, weight="semibold")

    #create the piechart legend
    pie_legend_ax = inset_axes(ax, width=largest_size, height=largest_size, loc='lower left', bbox_to_anchor=(0.06, 0.32, 1, 1), bbox_transform=ax.transAxes)
    dist_res = [8000000, 8000000]
    dist_ind = [10000000, 10000000]
    size_ind = max(calculate_size(sum(dist_ind), min_size, max_size, min_pie_size, max_pie_size, scale), min_pie_size)
    size_res = max(calculate_size(sum(dist_res), min_size, max_size, min_pie_size, max_pie_size, scale), min_pie_size)
    plot_splitted_piechart(dist_res, dist_ind, pie_legend_ax, size_res, size_ind, min_pie_size, labels=True, labels_text=['circular', 'rectangular', ''])

    pie_legend_ax.set_title("WWTP Type",fontsize=14,weight="semibold",y=1.05)
    pie_legend_ax.annotate('Residential', xy=(-0.1, 1), ha='left', va='center', fontsize=14, weight="semibold", xycoords='axes fraction')
    pie_legend_ax.annotate('Industrial', xy=(1.1, 1), ha='right', va='center', fontsize=14, weight="semibold", xycoords='axes fraction')
    pie_legend_ax.set_axis_off()

    # create title 
    ax.set_title("Worldwide Overview of WWTPs by Size and Technology",
                fontsize=24, fontweight='bold')
    try:
        plt.tight_layout()
    except Exception:
        # Inset/parasitic axes can occasionally fail tight layout on some backends.
        pass
    ensure_output_dir_for_file(cfg['paths']['static_piechart_filepath'])
    plt.savefig(cfg['paths']['static_piechart_filepath'], dpi=200)

if __name__ == '__main__':
    configure_logging()
    main()