"""Split Voronoi outputs into verification, non-verification, and single-site groups.

Watersheds are selected for verification when the share of valid WWTP shapes in a
basin exceeds the configured `percent_verification` threshold.
"""
import argparse
import os
import geopandas as gpd
from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from .. import provenance
from ..utils import atomic_output, configure_logging

def find_verification_watersheds(gdf, percent_verification, watershed_col='HYBAS_ID'):
    """Mark each row with verification flags derived from basin-level validity rates."""
    if watershed_col not in gdf.columns:
        raise KeyError(f"Missing watershed column '{watershed_col}'")
    if 'total_area' not in gdf.columns:
        raise KeyError("Missing required column 'total_area'")
    if not 0 <= float(percent_verification) <= 1:
        raise ValueError("percent_verification must be within [0, 1]")

    gdf = gdf.copy()
    gdf['is_single_points'] = (
        gdf.groupby(watershed_col)[watershed_col].transform('size') == 1
    )
    gdf['use_verify'] = (
        (~gdf['is_single_points'])
        & (gdf['total_area'] != 0.0)
        #& (gdf['round_area'] != 0)
    )
    gdf['watershed_fraction_valid'] = (
        gdf.groupby(watershed_col)['use_verify'].transform('mean')
    )
    gdf['watersheds_chosen'] = gdf['watershed_fraction_valid'] >= percent_verification
    return gdf


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run verification_script.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Split each population output into verification, non-verification, and single-site files."""
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="verification_script", **overrides)
    provenance.write_run_manifest(cfg, "verification_script")
    verification_dir = cfg['paths']['verification_dir']
    pop_dir = cfg['paths']['pop_output_dir']
    percent_verification = cfg['percent_verification']

    if not os.path.isdir(pop_dir):
        raise FileNotFoundError(f"Population output directory not found: {pop_dir}")

    if not os.path.exists(verification_dir):
        os.makedirs(verification_dir, exist_ok=True)
    filenames = [f for f in os.listdir(pop_dir) if f.endswith('.gpkg')]
    
    for filename in filenames:
        #if '_add' in filename:
        #    print(filename)
        #    continue
        filepath = os.path.join(pop_dir, filename)
        ver_output_filepath = os.path.join(verification_dir, f'ver_{filename}')
        unver_output_filepath = os.path.join(verification_dir, f'unver_{filename}')
        single_output_filepath = os.path.join(verification_dir, f'single_{filename}')

        gdf = gpd.read_file(filepath)
        gdf = find_verification_watersheds(gdf, percent_verification)

        ver_gdf = gdf[gdf['watersheds_chosen']].reset_index(drop=True)
        un_ver_gdf = gdf[(~gdf['watersheds_chosen']) & (~gdf['is_single_points'])].reset_index(drop=True)
        singles_df = gdf[gdf['is_single_points']].reset_index(drop=True)

        for part, path in ((ver_gdf, ver_output_filepath), (un_ver_gdf, unver_output_filepath),
                           (singles_df, single_output_filepath)):
            if not part.empty:
                with atomic_output(path) as tmp_path:
                    part.to_file(tmp_path, driver='GPKG', index=False)


if __name__ == '__main__':
    configure_logging()
    main()