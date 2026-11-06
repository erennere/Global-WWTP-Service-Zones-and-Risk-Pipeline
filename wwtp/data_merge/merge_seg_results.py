"""Merge segmentation outputs into WWTP geospatial datasets.

Supports two workflows:
- old: zipped tile CSVs plus mapping file
- new: single flat CSV keyed by image filename
"""

import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import zipfile
from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..utils import atomic_output, configure_logging
from ..geo_utils import nearest_within_threshold

def assign_to_nearest(gdf_source, gdf_target, threshold=None, one_to_one=False):
    """Attach nearest target attributes to each source geometry.

    Source rows without valid geometry or without a successful nearest-neighbor
    lookup are preserved and returned without merged target attributes.

    Parameters
    ----------
    gdf_source : geopandas.GeoDataFrame
        Source layer whose rows will receive nearest attributes.
    gdf_target : geopandas.GeoDataFrame
        Target layer queried through its spatial index.
    threshold : float | None, default=None
        Optional maximum nearest-neighbor distance in target CRS units.
        When ``None``, behavior is unchanged from the previous implementation.
    one_to_one : bool, default=False
        When true, each target is attached to at most one source row - the
        closest one (ties: the first). The other sources that picked the same
        target are returned unmatched, so a reference value can never be
        counted twice in a comparison.
    """
    gdf_source = gdf_source.copy()
    if gdf_source.crs is None or gdf_target.crs is None:
        raise ValueError("Both source and target GeoDataFrames must define a CRS")
    source_crs = gdf_source.crs
    gdf_source = gdf_source.to_crs(gdf_target.crs)
    sindex = gdf_target.sindex

    nearest_matches = [
        nearest_within_threshold(sindex, geom, threshold) for geom in gdf_source.geometry
    ]

    gdf_source['nearest_index'] = nearest_matches
    if one_to_one:
        matched = gdf_source['nearest_index'].notna()
        if matched.any():
            target_idx = gdf_source.loc[matched, 'nearest_index'].astype(int).to_numpy()
            distance = gdf_source.loc[matched].geometry.distance(
                gpd.GeoSeries(gdf_target.geometry.to_numpy()[target_idx], index=gdf_source.index[matched],
                              crs=gdf_target.crs)
            )
            order = pd.DataFrame({'target': target_idx, 'distance': distance.to_numpy(),
                                  'pos': np.arange(len(target_idx))}, index=gdf_source.index[matched])
            keep = order.sort_values(['target', 'distance', 'pos']).drop_duplicates('target').index
            losers = gdf_source.index[matched].difference(keep)
            gdf_source.loc[losers, 'nearest_index'] = None
    gdf_source_na = gdf_source[gdf_source['nearest_index'].isna()]
    gdf_source = gdf_source[gdf_source['nearest_index'].notna()].copy()
    gdf_source['nearest_index'] = gdf_source['nearest_index'].astype(int)
    gdf_source = gdf_source.merge(
        gdf_target, left_on='nearest_index', right_index=True, suffixes=('', '_nearest')
    )
    gdf_source = pd.concat([gdf_source, gdf_source_na], ignore_index=True)
    gdf_source = gpd.GeoDataFrame(gdf_source, geometry='geometry', crs=gdf_target.crs)

    if 'nearest_index' in gdf_source.columns:
        gdf_source.drop(columns=['nearest_index'], inplace=True)
    if 'geometry_nearest' in gdf_source.columns:
        gdf_source.drop(columns=['geometry_nearest'], inplace=True)
    return gdf_source.to_crs(source_crs)

def merge_old(cfg):
    """Merge zipped segmentation outputs into the legacy corrected dataset.

    Parameters
    ----------
    cfg : dict
        Runtime configuration dictionary.
    """
    paths = cfg['paths']
    mapping_filepath = paths["dl_mapfile"]
    zip_filepath = paths["dl_zipfile"]
    zip_output_path = os.path.join(paths['data_dir'], paths["dl_dir"], os.path.basename(paths["dl_zipfile"]).split('.')[-2])

    gdf = gpd.read_file(paths["corrected_south"])
    mapping = gpd.read_file(os.path.abspath(mapping_filepath))
    mapping['idx'] = mapping['idx'].astype(int)

    if not os.path.exists(zip_output_path):
        os.makedirs(zip_output_path, exist_ok=True)

    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(zip_output_path)

    data = []
    zip_output_path = os.path.join(zip_output_path, os.path.basename(paths["dl_zipfile"]).split('.')[-2])
    for file in [os.path.join(zip_output_path, f) for f in os.listdir(zip_output_path) if f.endswith('.csv')]:
        data.append(pd.read_csv(file))

    if not data:
        raise FileNotFoundError(f"No CSV segmentation files found in extracted archive path: {zip_output_path}")

    data = pd.concat(data, ignore_index=True)
    data['idx'] = data['File Name'].apply(lambda val: int(val.split('.')[0]))
    data['idx'] = data['idx'].astype(int)
    data = data.sort_values(by='idx', ascending=True)
    data = pd.merge(data, mapping, on=['idx'])
    data = gpd.GeoDataFrame(data, geometry='geometry', crs=mapping.crs)
    not_valids = gdf[gdf['geometry'].isna()]
    main_data = assign_to_nearest(gdf[gdf['geometry'].notna()], data)
    main_data = pd.concat([main_data, not_valids], ignore_index=True)
    output_path = os.path.join(paths["data_dir"], paths["seg_corrected_south"])
    with atomic_output(output_path) as tmp_path:
        main_data.to_file(tmp_path, driver='GPKG', index=False)

def merge_new(cfg):
    """Merge flat CSV segmentation outputs into the current corrected dataset.

    Parameters
    ----------
    cfg : dict
        Runtime configuration dictionary.
    """
    paths = cfg['paths']
    points_df = gpd.read_file(paths['corrected_all_filepath'])
    if 'idx' not in points_df.columns:
        raise KeyError("corrected_all dataset must include 'idx' column")
    points_df['idx'] = points_df['idx'].astype(int)
    seg_results = pd.read_csv(paths['seg_results_filepath'])
    if 'img_name' not in seg_results.columns:
        raise KeyError("seg_results file must include 'img_name' column")

    seg_results['idx'] = pd.to_numeric(
        seg_results['img_name'].astype(str).str.split('.').str[0],
        errors='coerce',
    )
    if seg_results['idx'].isna().any():
        raise ValueError("All seg_results img_name values must start with an integer id")
    seg_results['idx'] = seg_results['idx'].astype(int)

    cols = ['num_detection_circle', 'diameters', 'num_detection_rect', 'wwtp_area_rect']
    extra_cols = ['wwtp_area_square', 'num_detection_square']
    cols_to_drop = [col for col in cols + extra_cols if col in points_df.columns]
    points_df = points_df.drop(columns=cols_to_drop)

    merged_df = gpd.GeoDataFrame(
        pd.merge(points_df, seg_results, on='idx', how='left'),
        geometry='geometry',
        crs=points_df.crs,
    )
    with atomic_output(paths['corrected_all_filepath']) as tmp_path:
        merged_df.to_file(tmp_path, index=False, driver='GPKG')

def parse_args():
    """Parse command-line arguments for merge workflow selection."""
    parser = argparse.ArgumentParser(description='Merge segmentation outputs into geospatial datasets.')
    add_standard_override_arguments(parser)
    parser.add_argument(
        '--variant',
        choices=['old', 'new'],
        default='old',
        help='Choose the merge workflow: old (zipped tiles + mapping) or new (single CSV results).',
    )
    return parser.parse_args()

def main():
    """Parse CLI options and run the selected segmentation-merge workflow.

    Returns
    -------
    None
        This function dispatches to either the legacy or current merge path.
    """
    args = parse_args()
    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name="merge_seg_results", **overrides)

    if args.variant == 'old':
        if cfg['legacy_merge']:
            merge_old(cfg)
        return
    merge_new(cfg)

if __name__ == '__main__':
    configure_logging()
    main()