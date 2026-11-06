"""Build the final merged WWTP dataset from regional and model outputs.

This module merges corrected points with country-specific datasets,
applies confidence-aware enrichment, and deduplicates nearby facilities.
"""

import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Point, from_wkt
from .correct_locations_w_OSM import coordinate_corr_locations_wOSM
from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..utils import atomic_output, configure_logging
from ..geo_utils import (
    cluster_point_indices as _cluster_point_indices,
    estimate_utm_epsg_for_geom,
    parse_diameters_to_round_area,
    nearest_distances_m,
)

def cluster_point_indices(geoms, threshold):
    """Group point geometries (given as WKT strings) into connected components within a distance threshold."""
    return _cluster_point_indices([from_wkt(g) for g in geoms], threshold)

def cluster_points(df, threshold):
    """Aggregate records whose meter-space geometries fall into the same cluster."""
    df = df.copy()
    cluster_sets = cluster_point_indices(df['meter_geometry'], threshold)
    rows = []
    for cluster_set in cluster_sets:
        sub_df = df.iloc[list(cluster_set)].copy()
        if len(sub_df) == 1:
            rows.append(sub_df.iloc[0])
            continue

        # Keep the row with most non-null attributes as the representative.
        filled_counts = sub_df.notna().sum(axis=1)
        geom_idx = filled_counts.idxmax()
        merged = sub_df.apply(
            lambda col: col.dropna().iloc[0] if col.notna().any() else pd.NA
        )
        if 'POP_SERVED' in sub_df:
            # Records this close are one plant reported by several sources;
            # summing their populations would double-count it.
            merged['POP_SERVED'] = pd.to_numeric(sub_df['POP_SERVED'], errors='coerce').max()
        merged['n_merged_records'] = len(sub_df)
        if 'wwtp_area_square' in sub_df:
            merged['wwtp_area_square'] = '[' + str(sub_df['wwtp_area_square'].apply(
            lambda x: np.sum([
                float(i) for i in str(x).strip().strip('[]').split() 
                if i and i.lower() != 'none'
            ]) if pd.notnull(x) else 0).sum())+']'
        if 'diameters' in sub_df:
            sub_df['round_area'] = sub_df['diameters'].apply(parse_diameters_to_round_area)
            merged['round_area'] = sub_df['round_area'].sum()
        merged["geometry"] = df.loc[geom_idx, "geometry"]
        rows.append(merged)
    result = pd.DataFrame(rows).reset_index(drop=True)
    return result

def find_unmatched_targets(gdf_source, gdf_target, threshold):
    """Return target rows with no source feature within ``threshold`` metres.

    Distances are great-circle metres, whatever the inputs' CRS. Measuring in
    the target CRS (EPSG:4326 for the German GeoJSON) made ``threshold=500``
    mean 500 degrees, so every target counted as matched.
    """
    if gdf_target is None or gdf_target.empty:
        return gdf_target
    if gdf_source is None or gdf_source.empty:
        return gdf_target.copy()
    distances, _ = nearest_distances_m(gdf_source, gdf_target)
    return gdf_target[~(distances <= float(threshold))].copy()

def get_best_points(gdf):
    """Split points into high- and low-confidence sets after geometry override."""
    gdf = gdf.copy()
    gdf['geometry'] = gdf.apply(lambda row: Point(row['best_file2_lon'], row['best_file2_lat']) if pd.notna(row['best_file2_lon']) else row['geometry'], axis=1)
    high_conf =  gdf[gdf['detection_flag']].reset_index(drop=True)
    low_conf =  gdf[(gdf['detection_flag'] != True) & (pd.notna(gdf['geometry']))].reset_index(drop=True)
    return high_conf, low_conf

def find_safe_epsg(row):
    """Estimate a suitable projected EPSG code for distance-based operations."""
    return estimate_utm_epsg_for_geom(row['geometry'])
    
def find_meter_coordinates(df):
    """Create meter-space geometry WKT per EPSG group for clustering."""
    if df is None or df.empty:
        empty = df.copy() if df is not None else gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=4326)
        if 'meter_geometry' not in empty.columns:
            empty['meter_geometry'] = pd.Series(dtype=object)
        return gpd.GeoDataFrame(empty, crs=4326, geometry='geometry')

    if 'epsg' not in df.columns:
        raise KeyError("Missing required 'epsg' column")

    gdfs = []
    for epsg in df['epsg'].unique():
        if pd.isna(epsg):
            continue
        subdf = df[df['epsg'] == epsg].copy()
        if subdf.empty:
            continue
        subdf['meter_geometry'] = subdf.to_crs(epsg).geometry.apply(lambda g: g.wkt)
        gdfs.append(subdf)
    if not gdfs:
        out = df.copy()
        if 'meter_geometry' not in out.columns:
            out['meter_geometry'] = pd.Series(dtype=object)
        return gpd.GeoDataFrame(out, crs=4326, geometry='geometry')
    return gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs=4326,
        geometry='geometry'
    )

def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run final_data_merge.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Run final merge, confidence handling, OSM correction, and deduplication."""
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="final_data_merge", **overrides)
    paths = cfg['paths']
    osm_threshold = cfg['osm_threshold']
    threshold = cfg['threshold']

    # Start from the configured merged baseline; legacy_merge controls whether
    # segmentation-adjusted outputs are preferred over the directly corrected set.
    old_filepath = paths["seg_corrected_south"] if cfg['legacy_merge'] else paths["corrected_south"]
    old_df = gpd.read_file(old_filepath)

    # Add external country datasets that are maintained outside the core merge.
    canada_df = pd.read_csv(paths["canada_filepath"], encoding='latin1')
    canada_df['geometry'] = canada_df.apply(lambda row: Point(row['Longitude/ Longitude'], row['Latitude/ Latitude']), axis=1)
    canada_df = gpd.GeoDataFrame(canada_df, geometry='geometry', crs=4326)

    thailand_df = gpd.read_file(paths['thailand_filepath'])
    
    # US and EU sources contain confidence scores, so high- and low-confidence
    # records are handled separately before the final union.
    us_df = gpd.read_file(paths['us_new_filepath'])
    high_conf_us, low_conf_us = get_best_points(us_df)

    eu_df = gpd.read_file(paths['eu_new_filepath'])
    high_conf_eu, low_conf_eu = get_best_points(eu_df)

    high_conf = gpd.GeoDataFrame(pd.concat([high_conf_eu, high_conf_us], ignore_index=True), geometry='geometry', crs=4326)
    low_conf = gpd.GeoDataFrame(pd.concat([low_conf_eu, low_conf_us], ignore_index=True), geometry='geometry', crs=4326)

    merged_df = gpd.GeoDataFrame(pd.concat([old_df, high_conf, canada_df, thailand_df], axis=0, ignore_index=True), crs=4326, geometry='geometry')

    # Germany corrections are appended only for unmatched sites to avoid adding
    # duplicate facilities that are already covered by the merged baseline.
    germany_df = gpd.read_file(paths["germany_filepath"])
    germany_df['geometry'] = germany_df.apply(lambda row: Point(row['neigh_lon'],
                                                                 row['neigh_lat']) if pd.notna(row['neigh_lat'])
                                                                   else row['geometry'], axis=1)
    
    new_ones = find_unmatched_targets(merged_df, germany_df, threshold).to_crs(4326)
    merged_df = pd.concat([merged_df, new_ones], axis=0, ignore_index=True)
    merged_df = gpd.GeoDataFrame(merged_df, crs=4326, geometry='geometry')
    
    # Low-confidence sites are re-snapped against OSM features before inclusion.
    low_conf['epsg'] = low_conf.apply(find_safe_epsg, axis=1)
    pdf = gpd.read_file(paths["osmgeo_filepath"])
    pdf['epsg'] = pdf.apply(find_safe_epsg, axis=1)
    
    low_conf = coordinate_corr_locations_wOSM(osm_threshold, pdf, low_conf)
    low_conf['geometry'] = low_conf['matched_osm_geometry']
    low_conf = low_conf[low_conf['geometry'].notna()]

    merged_df = gpd.GeoDataFrame(pd.concat([merged_df, low_conf], axis=0, ignore_index=True), crs=4326, geometry='geometry')
    del low_conf, low_conf_eu,low_conf_us, high_conf, high_conf_eu, high_conf_us, new_ones, germany_df, canada_df, thailand_df, us_df, eu_df

    # Finally, cluster nearby points into single facilities while preserving the
    # richest attribute record and aggregating key numeric fields.
    none_geo_df = merged_df[merged_df.geometry.isna()].reset_index(drop=True)
    merged_df   = merged_df[merged_df.geometry.notna()].reset_index(drop=True)

    merged_df['epsg'] = merged_df.apply(find_safe_epsg, axis=1)
    merged_df = find_meter_coordinates(merged_df)
    merged_df = cluster_points(merged_df, threshold)
    merged_df = gpd.GeoDataFrame(pd.concat([merged_df, none_geo_df], axis=0, ignore_index=True), crs=4326, geometry='geometry')
    merged_df['idx'] = range(0, len(merged_df))

    if "FID" in merged_df:
        merged_df = merged_df.drop(columns=['FID'], errors='ignore')
    if "fid" in merged_df:
        merged_df = merged_df.drop(columns=['fid'], errors='ignore')

    rest = merged_df[~merged_df['geometry'].isin(set(old_df['geometry'].tolist()))].reset_index(drop=True)
    with atomic_output(paths["new_points_filepath"]) as tmp_path:
        rest.to_file(tmp_path, driver='GPKG', index=False)
    with atomic_output(paths["corrected_all_filepath"]) as tmp_path:
        merged_df.to_file(tmp_path, driver='GPKG', index=False)

if __name__ == '__main__':
    configure_logging()
    main()