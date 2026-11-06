"""Compute watershed minus served-population differences and annotate raster stats.

This script selects one population GeoPackage by index, computes geometric
differences against watersheds per UTM group, intersects the result with TIFF
data, and writes a diff GeoPackage.
"""

import os
import logging
import argparse
import pandas as pd
import geopandas as gpd
from shapely import from_wkt, to_wkt
from concurrent.futures import ProcessPoolExecutor, as_completed
from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from .. import provenance
from ..geo_utils import buffer_geometry, ensure_duckdb_spatial, estimate_utm_epsg_for_geom
from ..utils import atomic_output, configure_logging, duckdb_connection
from ..add_pop import intersect_all_files
from ..pipelines import create_pop_output_paths
from ..service_areas import intersects_with_country_db

logger = logging.getLogger(__name__)


def find_difference(watershed_gdf, pop_gdf, basin_col="HYBAS_ID"):
    """Compute geometric difference watershed - population geometry by basin ID.

    Parameters
    ----------
    watershed_gdf : geopandas.GeoDataFrame
        Watershed polygons for a selected EPSG subset.
    pop_gdf : geopandas.GeoDataFrame
        Population polygons for the same EPSG subset.
    basin_col : str, default='HYBAS_ID'
        Column name used to join watershed and population features.

    Returns
    -------
    pandas.DataFrame | None
        Difference dataframe with WKT converted back to shapely geometry.
    """
    watershed_local = watershed_gdf.copy()
    pop_local = pop_gdf.copy()

    watershed_local["geometry"] = watershed_local["geometry"].map(to_wkt)
    pop_local["geometry"] = pop_local["geometry"].map(to_wkt)
    try:
        with duckdb_connection() as conn:
            ensure_duckdb_spatial(conn)
            conn.register("watershed_gdf", watershed_local)
            conn.register("pop_gdf", pop_local)

            query = f"""
            -- REPLACE, not a second "geometry" column: DuckDB renames a duplicate
            -- to geometry_1, so the difference was dropped and the whole basin kept.
            SELECT a.* REPLACE (
                ST_AsText(ST_Difference(ST_GEOMFROMTEXT(a.geometry), ST_GEOMFROMTEXT(b.geometry))) AS geometry
            )
            FROM watershed_gdf AS a
            LEFT JOIN pop_gdf AS b
            ON a.{basin_col} = b.{basin_col}
            WHERE b.{basin_col} IS NOT NULL
            """
            df = conn.execute(query).df()
            df = df[df["geometry"].notna()].copy()
            df["geometry"] = df["geometry"].map(from_wkt)
            logger.info("Computed %s difference rows", len(df))
            return df
    except Exception as e:
        logger.exception("Error while computing differences: %s", e)
        return None


def process_epsg_group(epsg, watershed_gdf, pop_gdf, basin_col="HYBAS_ID"):
    """Process one EPSG bucket and return differences in EPSG:4326."""
    subset_pop_gdf = pop_gdf[pop_gdf["epsg"] == epsg]
    subset_watershed_gdf = watershed_gdf[watershed_gdf[basin_col].isin(subset_pop_gdf[basin_col].unique())]

    if subset_pop_gdf.empty or subset_watershed_gdf.empty:
        logger.info("EPSG %s skipped because one subset is empty", epsg)
        return None

    subset_pop_gdf = subset_pop_gdf.to_crs(epsg)
    subset_watershed_gdf = subset_watershed_gdf.to_crs(epsg)
    diff_gdf = find_difference(subset_watershed_gdf, subset_pop_gdf, basin_col=basin_col)
    if diff_gdf is not None:
        diff_gdf = gpd.GeoDataFrame(diff_gdf, geometry="geometry", crs=epsg).to_crs(4326)
    return diff_gdf


def find_differences(watershed_gdf, pop_gdf, max_workers=None, is_parallel=True, basin_col="HYBAS_ID"):
    """Compute, per basin, the part not covered by any service area.

    The service areas are dissolved per basin first. Subtracting each region
    from its basin separately gave one overlapping "difference" per plant, each
    still containing the other plants' regions, so a basin with k plants was
    counted k times and served areas were reported as unserved.
    """
    pop_local = pop_gdf[pop_gdf[basin_col].notna()][[basin_col, "geometry"]].dissolve(by=basin_col, as_index=False)
    pop_local["epsg"] = pop_local["geometry"].apply(estimate_utm_epsg_for_geom)
    gdf_list = []

    if is_parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_epsg_group, epsg, watershed_gdf, pop_local, basin_col): epsg
                for epsg in pop_local["epsg"].unique()
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    gdf_list.append(result)
    else:
        for epsg in pop_local["epsg"].unique():
            result = process_epsg_group(epsg, watershed_gdf, pop_local, basin_col)
            if result is not None:
                gdf_list.append(result)

    if gdf_list:
        gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), geometry="geometry", crs=4326)
        # Reprojected differences can self-touch; the country join's
        # ST_Intersection fails on such rings.
        gdf["geometry"] = gdf["geometry"].map(buffer_geometry)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
        logger.info("Total difference features: %s", len(gdf))
        return gdf
    return gpd.GeoDataFrame(columns=watershed_gdf.columns, geometry="geometry", crs=4326)


def parse_bool(value):
    """Convert common textual boolean values to bool."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_args():
    """Parse the parallel-execution flag and the standard config overrides."""
    parser = argparse.ArgumentParser(
        description="Compute population difference polygons for the population layer of figures.approach."
    )
    parser.add_argument(
        "--is-parallel",
        default="true",
        help="Whether to process EPSG groups in parallel (true/false)",
    )
    add_standard_override_arguments(parser)
    args = parser.parse_args()
    args.is_parallel = parse_bool(args.is_parallel)
    return args


def main():
    """Load config, select one population file, compute differences, and save output."""
    args = parse_args()
    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name="find_diff_pop", **overrides)
    provenance.write_run_manifest(cfg, "find_diff_pop")

    watershed_filepath = cfg["paths"]["watershed"]
    max_workers = cfg["max_workers"]
    tif_dir = cfg["paths"]["pop_tif_dir"]
    pop_dif_output_dir = cfg["paths"]["pop_dif_output_dir"]

    approach = str(cfg["figures"]["approach"])
    voronoi_map = create_pop_output_paths(cfg)["voronoi"]
    if approach not in voronoi_map:
        raise KeyError(
            f"Configured figures.approach '{approach}' is not available; expected one of {sorted(voronoi_map)}"
        )

    pop_input_path = os.path.abspath(voronoi_map[approach])
    if not os.path.exists(pop_input_path):
        raise FileNotFoundError(
            f"Population layer of approach {approach} not found (run add_pop first): {pop_input_path}"
        )
    filename = os.path.basename(pop_input_path)

    pop_gdf = gpd.read_file(pop_input_path)
    watershed_gdf = gpd.read_file(watershed_filepath)
    logger.info("Loaded %s population features and %s watershed features", len(pop_gdf), len(watershed_gdf))

    diff_gdf = find_differences(
        watershed_gdf,
        pop_gdf,
        max_workers=max_workers,
        is_parallel=args.is_parallel,
        basin_col=cfg["basin_column_name"],
    )
    country_col = cfg["country_output_column"]
    if country_col not in diff_gdf.columns:
        # HydroBASINS carries no country, so the uncovered parts get theirs here
        # (largest overlap); intersect_all_files needs it to pick the raster.
        if diff_gdf.empty:
            diff_gdf[country_col] = pd.Series(dtype=object)
        else:
            diff_gdf = intersects_with_country_db(
                diff_gdf,
                cfg["paths"]["overture"],
                polygon_country_col=cfg["country_boundary_column"],
                output_country_col=country_col,
            )
    diff_gdf = intersect_all_files(diff_gdf, tif_dir, max_workers=max_workers, country_col=country_col)
    logger.info("Post-intersection features: %s", len(diff_gdf))

    if not os.path.exists(pop_dif_output_dir):
        os.makedirs(pop_dif_output_dir, exist_ok=True)

    output_filepath = os.path.join(pop_dif_output_dir, f"diff_{filename}")
    with atomic_output(output_filepath) as tmp_path:
        diff_gdf.to_file(tmp_path, driver="GPKG", index=False)
    logger.info("Wrote output to %s", output_filepath)


if __name__ == "__main__":
    configure_logging()
    main()
