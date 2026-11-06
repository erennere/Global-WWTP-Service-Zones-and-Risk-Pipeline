"""Build an interactive Folium map summarizing WWTP mix and served population.

The map combines a choropleth, Voronoi polygon tooltips, and country-level pie
markers to provide a lightweight interactive overview for communication outputs.
"""

import argparse
import os
import geopandas as gpd
import pandas as pd
import folium
from branca.element import Template, MacroElement
import math

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..pipelines import create_pop_output_paths
from ..utils import configure_logging, ensure_output_dir_for_file, industrial_category_mask, resolve_latest_zonal_sum_column
from ._shared import SUM_ONLY, ensure_population_percentage_column
from . import _shared

def aggregate_by_country(gdf, country_column, agg_column, industrial_column=None, is_pop=False):
    """Aggregate per-facility metrics to country-level totals for mapping."""
    return _shared.aggregate_by_country(
        gdf, country_column, agg_column,
        industrial_column=industrial_column, is_pop=is_pop, stats=SUM_ONLY,
    )


def calculate_size(value, min_value, max_value, min_size, max_size):
    """Map a numeric value to a marker size within a configured range."""
    return _shared.calculate_size(
        value, min_value, max_value, min_size, max_size,
        scale='linear', degenerate='min', floor_nonpositive=True,
    )


def resolve_zonal_sum_column(df, preferred):
    """Resolve preferred zonal-sum column with fallback to latest available year."""
    return resolve_latest_zonal_sum_column(
        df,
        preferred,
        missing_message="No '*_zonal_sum' column found in population layer.",
    )[1]

def get_pie_svg(res_vals, ind_vals, size_px):
    """Return an inline SVG donut chart for one country marker.

    The left semicircle visualizes residential shares and the right semicircle
    visualizes industrial shares.
    """
    colors = ['#3182bd', '#9ecae1', '#e6550d', '#fdae6b'] 
    
    def polar_to_cartesian(cx, cy, r, angle_deg):
        """Convert polar coordinates on the marker circle into SVG coordinates."""
        return cx + r * math.cos(math.radians(angle_deg)), cy + r * math.sin(math.radians(angle_deg))
    
    def sector_path(start_deg, end_deg, color):
        """Build an SVG path for one donut-chart sector."""
        if abs(end_deg - start_deg) <= 0.1: return ""
        x1, y1 = polar_to_cartesian(50, 50, 45, start_deg)
        x2, y2 = polar_to_cartesian(50, 50, 45, end_deg)
        large_arc = 1 if abs(end_deg - start_deg) > 180 else 0
        return f'<path d="M 50 50 L {x1} {y1} A 45 45 0 {large_arc} 1 {x2} {y2} Z" fill="{color}" stroke="white" stroke-width="0.5"/>'
    
    res_t, ind_t = sum(res_vals) or 1e-9, sum(ind_vals) or 1e-9
    res_t1_split = 270 - (180 * (res_vals[0]/res_t))
    ind_t1_split = 270 + (180 * (ind_vals[0]/ind_t))
    return f'''<svg width="{size_px}" height="{size_px}" viewBox="0 0 100 100">
        {sector_path(res_t1_split, 270, colors[0])}{sector_path(90, res_t1_split, colors[1])}
        {sector_path(270, ind_t1_split, colors[2])}{sector_path(ind_t1_split, 450, colors[3])}
        <circle cx="50" cy="50" r="18" fill="white" />
    </svg>'''

def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run piechart_interactive.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Assemble the interactive served-population and WWTP-type overview map."""
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="piechart_interactive", **overrides)

    # Path Setup
    approach = str(cfg['figures']['approach'])
    pop_fp = os.path.abspath(create_pop_output_paths(cfg)['voronoi'][approach])
    boundaries_fp = cfg['paths']['country_boundaries_filepath']
    stats_fp = cfg['paths']['raster_country_stats_filepath']

    pop_col, ind_col = 'population_served_index', 'IND/RES'
    tag1, tag2, agg_t = 'round_area', 'wwtp_area_rect_2', 'sum'
    min_total_size = float(cfg['min_total_size'])

    # Load and Prepare Data
    boundaries = gpd.read_file(boundaries_fp).to_crs("EPSG:4326")
    boundaries['country'] = boundaries['ISO_A2_EH']
    pop_gdf = gpd.read_file(pop_fp).to_crs("EPSG:4326")
    pop_gdf['country'] = pop_gdf['ISO_2']
    agg_col = resolve_zonal_sum_column(pop_gdf, cfg['zonal_sum_default_column'])
    agg_year = agg_col.split('_')[0]
    alias = f'Pop ({agg_year}):'
    
    # Clean population for Voronoi tooltips
    pop_gdf[agg_col] = pop_gdf[agg_col].fillna(0).round(0)
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
    pop_gdf[ind_col] = industrial_mask

    pop_df_no_geom = pop_gdf.drop('geometry', axis=1)
    agg_ds = []
    for is_p, cols in {True: [agg_col], False: [tag1, tag2]}.items():
        for c in cols: agg_ds.append(aggregate_by_country(pop_df_no_geom, 'country', c, ind_col, is_p))
    
    for d in agg_ds:
        boundaries = boundaries.merge(d, on='country', how='left')

    if not os.path.exists(stats_fp):
        raise FileNotFoundError(f"Stats file not found: {stats_fp}")
    stats_df = pd.read_csv(stats_fp)
    stats_df.columns = [str(c).strip().lstrip('\ufeff') for c in stats_df.columns]
    boundaries = boundaries.merge(stats_df, on='country', how='left')
    
    boundaries['total_size'] = boundaries[[f'IND_{tag1}_{agg_t}', f'IND_{tag2}_{agg_t}', 
                                         f'RES_{tag1}_{agg_t}', f'RES_{tag2}_{agg_t}']].sum(axis=1)
    pop_col = ensure_population_percentage_column(boundaries, pop_col, agg_col)
    boundaries[pop_col] = pd.to_numeric(boundaries[pop_col], errors='coerce').fillna(0) * 100
    
    m = folium.Map(location=[10, 0], zoom_start=3, tiles='CartoDB positron')

    # 1. Choropleth
    choro = folium.Choropleth(
        geo_data=boundaries.to_json(), data=boundaries, columns=['country', pop_col],
        key_on="feature.properties.country", fill_color='YlGnBu', fill_opacity=0.6,
        line_opacity=0.2, legend_name="Percentage of People Served by a WWTP"
    ).add_to(m)

    # 2. Voronoi (Rounded Pop Tooltip)
    choro_geojson_name = choro.geojson.get_name()
    pop_gdf['geometry'] = pop_gdf.geometry.simplify(0.005)
    voronoi_layer = folium.GeoJson(
        pop_gdf[['geometry', agg_col, ind_col]],
        name="Voronoi Details",
        style_function=lambda x: {
            'fillColor': '#e6550d' if x['properties'][ind_col] else '#3182bd',
            'color': 'white', 'weight': 0.4, 'fillOpacity': 0.4
        },
        highlight_function=lambda x: {'weight': 2, 'color': 'black', 'fillOpacity': 0.7},
        tooltip=folium.GeoJsonTooltip(
            fields=[agg_col], 
            aliases=[alias], 
            localize=True, # Adds thousand separators
            sticky=True
        )
    ).add_to(m)

    # 3. Pie Markers Group
    pie_group = folium.FeatureGroup(name="Country Pies").add_to(m)
    v_min, v_max = boundaries['total_size'].min(), boundaries['total_size'].max()
    
    for _, row in boundaries.iterrows():
        if pd.isna(row.geometry) or row['total_size'] < min_total_size:
            continue
        
        pos = row.geometry.centroid if row.geometry.geom_type == 'Polygon' else max(list(row.geometry.geoms), key=lambda x: x.area).centroid
        s_px = int(calculate_size(row['total_size'], v_min, v_max, 25, 85))
        
        # Values in km2 for popup
        total_km2 = row['total_size'] / 1_000_000
        r1, r2 = (row.get(f'RES_{tag1}_{agg_t}', 0)/1e6), (row.get(f'RES_{tag2}_{agg_t}', 0)/1e6)
        i1, i2 = (row.get(f'IND_{tag1}_{agg_t}', 0)/1e6), (row.get(f'IND_{tag2}_{agg_t}', 0)/1e6)
        
        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 11px; width: 220px;">
            <b style="font-size: 13px;">{row['country']}</b><br>
            <b>Total WWTP Area:</b> {total_km2:,.2f} km&sup2;<br>
            <b>Pop (2024):</b> {row.get('population_total', 0):,.0f}<br><hr style="margin:4px 0;">
            <b>Pop Served:</b> {row.get('population_served', 0):,.0f}<br>
            <b>Pop Served [%]:</b> {row.get(pop_col, 0):,.1f}%<br><hr style="margin:4px 0;">
            <span style="color:#3182bd;">&#9679;</span> Res Circular: {r1:,.2f} km&sup2;<br>
            <span style="color:#9ecae1;">&#9632;</span> Res Rect: {r2:,.2f} km&sup2;<br>
            <span style="color:#e6550d;">&#9679;</span> Ind Circular: {i1:,.2f} km&sup2;<br>
            <span style="color:#fdae6b;">&#9632;</span> Ind Rect: {i2:,.2f} km&sup2;
        </div>
        """
        folium.Marker(
            [pos.y, pos.x], 
            icon=folium.DivIcon(
                html=get_pie_svg([r1, r2], [i1, i2], s_px),
                icon_size=(s_px, s_px), icon_anchor=(s_px/2, s_px/2)
            ),
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(pie_group)

    # 4. JavaScript Layer Toggling
    m.get_root().html.add_child(folium.Element(f"""
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                var map = {m.get_name()};
                function updateView() {{
                    var zoom = map.getZoom();
                    var vLayer = {voronoi_layer.get_name()};
                    var pLayer = {pie_group.get_name()};
                    var choroGeoJson = {choro_geojson_name};
                    if (zoom >= 6) {{
                        if (!map.hasLayer(vLayer)) {{ map.addLayer(vLayer); }}
                        if (map.hasLayer(pLayer)) {{ map.removeLayer(pLayer); }}
                        choroGeoJson.setStyle({{fillOpacity: 0.1, opacity: 0.1}});
                    }} else {{
                        if (map.hasLayer(vLayer)) {{ map.removeLayer(vLayer); }}
                        if (!map.hasLayer(pLayer)) {{ map.addLayer(pLayer); }}
                        choroGeoJson.setStyle({{fillOpacity: 0.6, opacity: 0.2}});
                    }}
                }}
                map.on('zoomend', updateView);
                setTimeout(updateView, 400); 
            }});
        </script>
    """))

    # 5. CLEAN LEGEND SCALE
    # Find nice rounded numbers for the legend based on max value
    max_km2_actual = v_max / 1_000_000
    if max_km2_actual > 100: leg_high = math.ceil(max_km2_actual / 50) * 50
    elif max_km2_actual > 10: leg_high = math.ceil(max_km2_actual / 5) * 5
    else: leg_high = math.ceil(max_km2_actual)
    
    leg_mid = leg_high / 2
    leg_low = leg_high / 10 if leg_high > 1 else 0.1

    legend_html = f'''
    {{% macro html(this, kwargs) %}}
    <div style="position: fixed; bottom: 30px; left: 30px; width: 260px; z-index: 9999; 
                background: white; border-radius: 8px; padding: 15px; font-family: sans-serif;
                box-shadow: 0 0 10px rgba(0,0,0,0.3); border: 1px solid #ccc;">
        <div style="font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid #eee;">WWTP Type Breakdown</div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <svg width="45" height="45" viewBox="0 0 100 100">
                <path d="M 50 50 L 50 10 A 40 40 0 0 0 50 90 Z" fill="#3182bd" />
                <path d="M 50 50 L 50 10 A 40 40 0 0 1 50 90 Z" fill="#e6550d" />
                <circle cx="50" cy="50" r="18" fill="white" />
            </svg>
            <div style="font-size: 11px;">
                <b>Residential (Left)</b><br>
                <b>Industrial (Right)</b><br>
                <span style="color:#555;">Dark: Circular | Light: Rectangular</span>
            </div>
        </div>
        <div style="font-weight: bold; margin-top: 15px; margin-bottom: 8px; border-top: 1px solid #eee; padding-top: 8px;">Total Area Scale</div>
        <div style="position: relative; height: 70px;">
            <div style="position: absolute; bottom: 0; width: 60px; height: 60px; border: 1.5px solid #444; border-radius: 50%;"></div>
            <div style="position: absolute; bottom: 0; left: 10px; width: 40px; height: 40px; border: 1.5px solid #444; border-radius: 50%;"></div>
            <div style="position: absolute; bottom: 0; left: 20px; width: 20px; height: 20px; border: 1.5px solid #444; border-radius: 50%;"></div>
            <div style="position: absolute; left: 75px; bottom: 48px; font-size: 11px;">&mdash; {leg_high:,.0f} km&sup2;</div>
            <div style="position: absolute; left: 75px; bottom: 28px; font-size: 11px;">&mdash; {leg_mid:,.1f} km&sup2;</div>
            <div style="position: absolute; left: 75px; bottom: 8px; font-size: 11px;">&mdash; {leg_low:,.1f} km&sup2;</div>
        </div>
    </div>
    {{% endmacro %}}
    '''
    macro = MacroElement()
    macro._template = Template(legend_html)
    m.get_root().add_child(macro)
    
    ensure_output_dir_for_file(cfg['paths']['interactive_piechart_html_filepath'])
    m.save(cfg['paths']['interactive_piechart_html_filepath'])

if __name__ == '__main__':
    configure_logging()
    main()