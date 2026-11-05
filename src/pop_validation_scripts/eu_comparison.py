"""Compare WWTP-derived served-population estimates against the EU reference layer.

The script joins project outputs to the UWWTD reference dataset, computes both
normalized difference and multiplicative comparison metrics, and exports yearly
histogram panels for verification subsets.
"""
import argparse
import os
import logging
import geopandas as gpd

try:
    from .hw_comparison import extract_voronoi_parameters, orchestrate_single
    from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
    from ..utils import configure_logging
    from ..geo_utils import load_eu_reference_layer
    from ..data_merge.merge_seg_results import assign_to_nearest
except ImportError:
    from src.pop_validation_scripts.hw_comparison import extract_voronoi_parameters, orchestrate_single
    from src.starter import add_standard_override_arguments, load_config, parse_config_overrides
    from src.utils import configure_logging
    from src.geo_utils import load_eu_reference_layer
    from src.data_merge.merge_seg_results import assign_to_nearest

logger = logging.getLogger(__name__)

def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run eu_comparison.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Load verification files, align them to the EU reference layer, and plot comparisons.

    Returns
    -------
    None
        The function iterates over configured verification files and writes the
        generated comparison plots to disk.
    """
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="eu_comparison", **overrides)
    ver_dir = cfg['paths']['verification_dir']
    plots_dir = cfg['paths']['eu_plots_dir']
    if not os.path.isdir(ver_dir):
        logger.warning("Verification directory not found: %s", ver_dir)
        return

    pop_filepaths = [os.path.join(ver_dir, f) for f in os.listdir(ver_dir) if f.endswith('.gpkg')]
    plot_args = {
    'dpi' : 300,
    'bins' : 100,
    'save' : True,
    'fontsize' : 26,
    'small_fontsize' : 18,
    'lower_quantile' : 0.01}

    threshold = cfg['threshold']
    pop_col = 'POP_SERVED_EU'
    ref_filepath = cfg['paths']['eu_ref_filepath']
    organic_m_column = 'uwwCapacity'

    ref_file = load_eu_reference_layer(
        ref_filepath,
        cfg['eu_reference_factor'],
        capacity_column=organic_m_column,
        pop_column=pop_col,
    )

    for filepath in pop_filepaths:
        filename = os.path.basename(filepath)
        params = extract_voronoi_parameters(filepath)
        approach = params.get('approach') if isinstance(params, dict) else None
        gdf = gpd.read_file(filepath)
        gdf = assign_to_nearest(gdf, ref_file, threshold, one_to_one=True)
        gdf = gdf[gdf[organic_m_column].notna()].reset_index(drop=True)
        orchestrate_single(
            gdf, approach, plot_args, plots_dir, filename, pop_col,
            qual_pop_default=None,
            filter_qual_pop=False,
            upper_quantile_hw_comp=0.95,
            comp_output_prefix='eu_comp',
            reference_name='Reference EU',
            hide_empty_axis=False,
        )

if __name__ == '__main__':
    configure_logging()
    main()