"""Generate lightweight GeoJSON and HTML for the sizes interactive map.

The script reads the configured default Voronoi layer, keeps only the columns
needed by frontend JavaScript, converts polygons to centroid points, and writes
a standalone HTML output with the GeoJSON fetch path wired to the generated file.
"""

import argparse
import os
import re
from typing import Optional

import geopandas as gpd
import pandas as pd

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..pipelines import create_pop_output_paths
from ..utils import configure_logging, ensure_output_dir_for_file, resolve_latest_zonal_sum_column


def _resolve_population_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """Return best-available population-like column name from input layer."""
    # Lexicographic, not year-ranked: preserved deliberately so the column this
    # map labels does not change (see utils.resolve_latest_zonal_sum_column).
    _, zonal_col = resolve_latest_zonal_sum_column(gdf, lexicographic=True, required=False)
    if zonal_col is not None:
        return zonal_col

    for candidate in ("POP_SERVED", "population_served", "pop_served", "population", "pop_sum"):
        if candidate in gdf.columns:
            return candidate
    return None


def _build_geojson(cfg: dict) -> str:
    """Create reduced GeoJSON with polygon geometry, centroid metadata, and bbox index fields."""
    approach = str(cfg["figures"]["approach"])
    pop_fp = os.path.abspath(create_pop_output_paths(cfg)["voronoi"][approach])

    gdf = gpd.read_file(pop_fp)
    pop_col = _resolve_population_column(gdf)
    if pop_col is None:
        raise KeyError(
            "No population column found in source layer. Expected latest '*_zonal_sum' or one of "
            "['POP_SERVED', 'population_served', 'pop_served', 'population', 'pop_sum']."
        )

    keep_cols = ["geometry", "total_area", "round_area", pop_col]
    gdf = gdf[keep_cols].copy()
    gdf = gdf.rename(columns={pop_col: "population"})
    gdf["population"] = pd.to_numeric(gdf["population"], errors="coerce").fillna(0)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Preserve polygon geometry for high-zoom rendering in the HTML map.
    # Store representative points for low-zoom marker clustering.
    reps = gdf.geometry.representative_point()
    gdf["centroid_lon"] = reps.x
    gdf["centroid_lat"] = reps.y

    # Precompute bbox columns so the frontend can cull polygons without
    # constructing temporary layers for bounds calculation.
    bounds = gdf.geometry.bounds
    gdf["bbox_minx"] = bounds.minx
    gdf["bbox_miny"] = bounds.miny
    gdf["bbox_maxx"] = bounds.maxx
    gdf["bbox_maxy"] = bounds.maxy

    geojson_out = cfg["paths"]["sizes_interactive_geojson_filepath"]
    ensure_output_dir_for_file(geojson_out)
    gdf.to_file(geojson_out, driver="GeoJSON", index=False)
    return geojson_out


def _build_html(cfg: dict, geojson_out: str) -> str:
    """Create HTML map from template and inject relative GeoJSON fetch path."""
    template_fp = cfg["paths"]["sizes_interactive_template_filepath"]
    html_out = cfg["paths"]["sizes_interactive_html_filepath"]

    with open(template_fp, "r", encoding="utf-8") as file:
        html = file.read()

    rel_geojson = os.path.relpath(geojson_out, start=os.path.dirname(html_out)).replace("\\", "/")
    if not rel_geojson.startswith("."):
        rel_geojson = f"./{rel_geojson}"

    updated_html = re.sub(r'fetch\("[^"]+"\)', f'fetch("{rel_geojson}")', html, count=1)

    ensure_output_dir_for_file(html_out)
    with open(html_out, "w", encoding="utf-8") as file:
        file.write(updated_html)

    return html_out


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run sizes_interactive_map.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Build reduced GeoJSON and write a versioned sizes interactive HTML map."""
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="sizes_interactive_map", **overrides)

    geojson_out = _build_geojson(cfg)
    html_out = _build_html(cfg, geojson_out)

    print(f"GeoJSON written: {geojson_out}")
    print(f"HTML map written: {html_out}")


if __name__ == "__main__":
    configure_logging()
    main()
