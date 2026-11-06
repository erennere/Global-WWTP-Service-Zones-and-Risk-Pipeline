"""Assign each river segment to the most representative HYBAS basin polygon.

The script spatially intersects river lines with watershed polygons, resolves
ambiguous line-to-polygon matches by longest overlap, and writes enriched river
features to the configured output path. It expects HydroRIVERS-style river IDs
(`HYRIV_ID`) and HydroBASINS basin IDs (`HYBAS_ID`).
"""

import argparse
import logging
import geopandas as gpd
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from .. import provenance
from ..utils import atomic_output, configure_logging

logger = logging.getLogger(__name__)


def assign_hybas_id_by_length(lines_gdf, poly_gdf, id_col="HYBAS_ID"):
    """Assign basin IDs to lines using maximum intersection length.

    Parameters
    ----------
    lines_gdf : geopandas.GeoDataFrame
        River line features.
    poly_gdf : geopandas.GeoDataFrame
        Watershed polygons containing `id_col`.
    id_col : str, default='HYBAS_ID'
        Polygon identifier column to transfer to lines.

    Returns
    -------
    geopandas.GeoDataFrame
        Input lines with assigned basin IDs.
    """
    if id_col not in poly_gdf.columns:
        raise KeyError(f"Polygon column '{id_col}' not found")

    if lines_gdf.empty or poly_gdf.empty:
        logger.info("Skipping assignment because one input GeoDataFrame is empty")
        return lines_gdf

    # 1. Ensure CRS match
    if lines_gdf.crs != poly_gdf.crs:
        lines_gdf = lines_gdf.to_crs(poly_gdf.crs)

    # 2. Add temporary unique ID
    lines_gdf["_tmp_id"] = range(len(lines_gdf))

    # 3. Fast Spatial Join to find potential matches
    potential_matches = gpd.sjoin(
        lines_gdf[["_tmp_id", "geometry"]], poly_gdf[[id_col, "geometry"]], how="inner", predicate="intersects"
    )
    if potential_matches.empty:
        lines_gdf[id_col] = None
        return lines_gdf.drop(columns=["_tmp_id"])

    counts = potential_matches["_tmp_id"].value_counts()
    single_match_ids = counts[counts == 1].index
    multi_match_ids = counts[counts > 1].index

    # 4. Handle Single Matches
    single_matches = potential_matches[potential_matches["_tmp_id"].isin(single_match_ids)]
    results_map = single_matches.set_index("_tmp_id")[id_col].to_dict()

    # 5. Handle Multi-Matches (The "Heavy" Path)
    if not multi_match_ids.empty:
        multi_lines = lines_gdf[lines_gdf["_tmp_id"].isin(multi_match_ids)]

        # Intersect lines with polygons
        fragments = gpd.overlay(multi_lines, poly_gdf[[id_col, "geometry"]], how="intersection")

        # CRITICAL: Project to Equal Area (meters) for accurate length comparison
        # We use World Cylindrical Equal Area (EPSG:54034) or similar
        fragments["len"] = fragments.to_crs(epsg=3857).geometry.length

        winners = fragments.sort_values("len", ascending=False).drop_duplicates("_tmp_id")
        results_map.update(winners.set_index("_tmp_id")[id_col].to_dict())

    # 6. Final Assignment
    lines_gdf[id_col] = lines_gdf["_tmp_id"].map(results_map)
    return lines_gdf.drop(columns=["_tmp_id"])


def extract_first_digit(df, source_col, new_col="first_digit"):
    """Extract the first character of `source_col` into `new_col`."""
    if source_col not in df.columns:
        raise KeyError(f"Missing source column '{source_col}'")

    df[new_col] = df[source_col].astype("string").str.strip().str[0]
    return df


def orchestrate_intersections(hybas_gdf, rivers_gdf, hybas_col, hyshed_col, new_col, max_workers=2):
    """Run per-region basin assignment in parallel and concatenate results.

    Parameters
    ----------
    hybas_gdf : geopandas.GeoDataFrame
        Watershed polygons.
    rivers_gdf : geopandas.GeoDataFrame
        River segments to enrich with basin IDs.
    hybas_col : str
        Basin identifier column in ``hybas_gdf``.
    hyshed_col : str
        River identifier column used to derive the grouping region.
    new_col : str
        Temporary grouping column created from the leading identifier digit.
    max_workers : int, default=2
        Maximum number of worker processes.

    Returns
    -------
    geopandas.GeoDataFrame
        River segments with assigned basin IDs.
    """
    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")

    # 1. Extract digits
    rivers_gdf = extract_first_digit(rivers_gdf, hyshed_col, new_col)
    hybas_gdf = extract_first_digit(hybas_gdf, hybas_col, new_col)

    continents = rivers_gdf[new_col].dropna().unique().tolist()
    gdfs = []
    discarded_regions = []
    failed_regions = []

    logger.info("Submitting tasks for %s regions", len(continents))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for c in continents:
            # Filter and copy to minimize pickle size
            r_chunk = rivers_gdf[rivers_gdf[new_col] == c].copy()
            h_chunk = hybas_gdf[hybas_gdf[new_col] == c].copy()

            if h_chunk.empty:
                # Rivers with no basin to assign are dropped rather than passed
                # through unmodified - an unassigned river is not a result.
                logger.warning(
                    "Region %s: no matching basin polygons; discarding %s river segments.",
                    c,
                    len(r_chunk),
                )
                discarded_regions.append(c)
                continue

            futures[executor.submit(assign_hybas_id_by_length, r_chunk, h_chunk, hybas_col)] = c

        # Adding tqdm progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Regions"):
            continent_code = futures[future]
            try:
                result = future.result()
                gdfs.append(result)
            except Exception as err:
                logger.exception("Region %s failed: %s", continent_code, err)
                failed_regions.append(continent_code)

    if not gdfs:
        raise RuntimeError(
            "Basin assignment produced no assigned rivers: "
            f"{len(failed_regions)} region(s) failed ({', '.join(map(str, failed_regions)) or 'none'}), "
            f"{len(discarded_regions)} region(s) had no matching basins "
            f"({', '.join(map(str, discarded_regions)) or 'none'})."
        )
    if failed_regions:
        logger.error(
            "Basin assignment incomplete: %s of %s submitted region(s) failed (%s).",
            len(failed_regions),
            len(failed_regions) + len(gdfs),
            ", ".join(map(str, failed_regions)),
        )
    logger.info("Completed intersections. Combining %s region outputs", len(gdfs))
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=rivers_gdf.crs)


def main():
    """Load config, assign basin IDs to rivers, and write output GeoPackage.

    CLI usage:
        python -m ...assign_rivers_to_basin --max-workers 2 [--level ... --version ...]
    """

    parser = argparse.ArgumentParser(description="Assign river segments to HydroBASINS polygons.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum worker processes for per-region assignment",
    )
    add_standard_override_arguments(parser)
    args = parser.parse_args()

    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name="assign_rivers_to_basin", **overrides)
    provenance.write_run_manifest(cfg, "assign_rivers_to_basin")

    hyshed_col = "HYRIV_ID"
    hybas_col = "HYBAS_ID"
    max_workers = int(args.max_workers)
    new_col = "continent"

    poly_path = cfg["paths"]["watershed"]
    line_path = cfg["paths"]["rivershed"]
    output_path = cfg["paths"]["rivershed_output_path"]

    logger.info("Reading input files")
    hybas_gdf = gpd.read_file(poly_path)
    rivers_gdf = gpd.read_file(line_path)
    logger.info("Loaded %s watersheds and %s river segments", len(hybas_gdf), len(rivers_gdf))

    final_rivers = orchestrate_intersections(
        hybas_gdf, rivers_gdf, hybas_col, hyshed_col, new_col, max_workers=max_workers
    )

    logger.info("Saving %s results to %s", len(final_rivers), output_path)
    with atomic_output(output_path) as tmp_path:
        final_rivers.to_file(tmp_path, driver="GPKG", index=False)
    logger.info("Process complete")


if __name__ == "__main__":
    configure_logging()
    main()
