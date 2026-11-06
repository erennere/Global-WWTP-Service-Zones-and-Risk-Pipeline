"""Population Data Integration Module

Intersects population raster data (GeoTIFFs) with geospatial vector data (Voronoi diagrams)
and aggregates population statistics using zonal statistics.

Workflow:
1. Load Voronoi polygon layer from GeoPackage
2. Locate corresponding population raster tiles by country (ISO code lookup)
3. Compute zonal statistics (sum, std) for population within each polygon
4. Add year-specific population columns to original geodataframe
5. Export enhanced geodataframe to GeoPackage with '_pop_added_' prefix

Supports parallel processing across multiple countries and years of population data.
Requires rasterized population TIFFs organized by ISO-3 country codes.
"""

import os
import logging
import sys
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union
from shapely.validation import explain_validity
from .starter import load_config, parse_config_overrides, add_standard_override_arguments
from . import provenance
from .pipelines import APPROACHES, create_output_paths, create_pop_output_paths
from .utils import atomic_output, configure_logging, get_iso_codes
from .geo_utils import repair_geometry
from exactextract import exact_extract

logger = logging.getLogger(__name__)

# Configure logging


def _normalize_country_code(value):
    """Normalize ISO-2 codes for robust joins against raster directory naming."""
    if pd.isna(value):
        return None
    code = str(value).strip().upper()
    return code if code else None


def _polygonal_repair(geom):
    """Repair an invalid polygon and keep only its polygonal parts (or ``None``)."""
    repaired = repair_geometry(geom)
    if repaired is None or repaired.geom_type in ("Polygon", "MultiPolygon"):
        return repaired
    parts = [g for g in getattr(repaired, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts) if parts else None


def _find_failing_geometry_indices(tif_path, gdf, max_report=10):
    """Locate feature indices that still trigger exact_extract parse errors.

    Uses recursive chunking to avoid O(n) single-feature checks on large countries.
    """
    if gdf is None or gdf.empty:
        return []

    candidate_indices = list(gdf.index)
    failing = []
    stack = [candidate_indices]

    while stack and len(failing) < max_report:
        idxs = stack.pop()
        if not idxs:
            continue

        subset = gdf.loc[idxs]
        try:
            exact_extract(rast=tif_path, vec=subset, ops=["sum"], output="pandas")
            continue
        except Exception:
            if len(idxs) == 1:
                failing.append(idxs[0])
            else:
                mid = len(idxs) // 2
                stack.append(idxs[:mid])
                stack.append(idxs[mid:])

    return failing


def intersect_single_file(gdf, tif_paths, all_years=True, years=None):
    """Compute zonal statistics of population rasters within polygons using exactextract.

    Args:
        gdf: GeoDataFrame with polygon geometries
        tif_paths: List of population raster file paths (single or multiple years)
        all_years: When false, only process the most recent year found in tif_paths.
        years: Raster years to use (others are skipped); None uses every year found.
    Returns:
        GeoDataFrame: Input GeoDataFrame with added columns for year-specific population stats
    """
    if gdf is None or gdf.empty:
        return gdf

    country_hint = None
    if "ISO_2" in gdf.columns:
        country_values = gdf["ISO_2"].dropna().astype(str).unique().tolist()
        if len(country_values) == 1:
            country_hint = country_values[0]
        elif country_values:
            country_hint = ",".join(country_values[:3])

    org_crs = gdf.crs

    # Map raster files to the year encoded in their filenames.
    my_dict = {}
    for file in tif_paths:
        if not os.path.exists(file):
            logger.warning(f"Raster file does not exist: {file}")
            continue

        basename = os.path.basename(file)
        parts = [int(k) for k in basename.split("_") if k.startswith("20") and len(k) == 4]

        if parts and years is not None and parts[0] not in years:
            logger.info("Skipping %s: year %s is not configured (download_pop years)", basename, parts[0])
        elif parts:
            my_dict[parts[0]] = file
        else:
            logger.warning(f"Could not extract year from filename: {basename}")

    if not my_dict:
        logger.warning("No usable raster years were found for the provided TIFF paths")
        return gdf

    # Process each raster year and attach exactextract outputs back onto gdf.
    last_year = max(my_dict.keys())
    for year, tif_path in my_dict.items():
        if not all_years and year != last_year:
            continue
        try:
            # Reproject polygons to raster CRS before extracting zonal statistics.
            with rasterio.open(tif_path) as src:
                raster_crs = src.crs

            if raster_crs is not None and gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)

            # exact_extract fails on empty/null/invalid geometries; drop them from extraction,
            # then set their stats to zero to keep row cardinality stable.
            geom = gdf.geometry
            null_mask = geom.isna()
            empty_mask = geom.is_empty.fillna(False)
            invalid_mask = ~geom.is_valid.fillna(False)
            bad_mask = null_mask | empty_mask | invalid_mask

            if bad_mask.any():
                bad_indices = gdf.index[bad_mask].tolist()
                sample_idx = bad_indices[:10]
                invalid_sample = []
                for idx in gdf.index[invalid_mask][:5].tolist():
                    try:
                        reason = explain_validity(gdf.at[idx, "geometry"])
                    except Exception:
                        reason = "unknown"
                    invalid_sample.append((idx, reason))

                logger.warning(
                    "Dropping %s geometries before exact_extract for country=%s year=%s (null=%s empty=%s invalid=%s sample_indices=%s invalid_sample=%s)",
                    int(bad_mask.sum()),
                    country_hint,
                    year,
                    int(null_mask.sum()),
                    int(empty_mask.sum()),
                    int(invalid_mask.sum()),
                    sample_idx,
                    invalid_sample,
                )

            # Invalid polygons are repaired for the extraction instead of being
            # counted as zero people; null geometries get NaN (not measured),
            # empty ones a true zero.
            skip_mask = null_mask | empty_mask
            extract_gdf = gdf.loc[~skip_mask].copy()
            repair_rows = invalid_mask[~skip_mask]
            if repair_rows.any():
                extract_gdf.loc[repair_rows, "geometry"] = extract_gdf.loc[repair_rows, "geometry"].map(
                    _polygonal_repair
                )
                extract_gdf = extract_gdf[extract_gdf.geometry.notna()]
            if extract_gdf.empty:
                raise RuntimeError(
                    f"No valid geometries remain for exact_extract for country={country_hint}, year={year}, tif={tif_path}"
                )

            # exact_extract is the active zonal-statistics backend for this module.
            stats_df = exact_extract(
                rast=tif_path,
                vec=extract_gdf,
                ops=["sum", "stdev"],
                output="pandas",  # Returns a tidy dataframe
            )

            # exact_extract returns canonical 'sum' and 'stdev' columns.
            sum_col = f"{year}_zonal_sum"
            std_col = f"{year}_zonal_std"
            gdf[sum_col] = 0.0
            gdf[std_col] = 0.0
            not_measured = gdf.index.difference(extract_gdf.index).difference(gdf.index[empty_mask])
            gdf.loc[not_measured, [sum_col, std_col]] = np.nan
            gdf.loc[extract_gdf.index, sum_col] = stats_df["sum"].clip(lower=0).values
            gdf.loc[extract_gdf.index, std_col] = stats_df["stdev"].clip(lower=0).values

            logger.info(f"Processed population for year {year}")

        except Exception as err:
            geometry_col = gdf.geometry if "geometry" in gdf.columns else None
            null_geom_count = int(geometry_col.isna().sum()) if geometry_col is not None else -1
            empty_geom_count = int(geometry_col.is_empty.fillna(False).sum()) if geometry_col is not None else -1
            invalid_geom_count = int((~geometry_col.is_valid.fillna(False)).sum()) if geometry_col is not None else -1
            geom_type_counts = (
                geometry_col.geom_type.value_counts(dropna=False).head(10).to_dict() if geometry_col is not None else {}
            )

            raster_exists = os.path.exists(tif_path)
            raster_meta = {}
            if raster_exists:
                try:
                    with rasterio.open(tif_path) as src_meta:
                        raster_meta = {
                            "crs": str(src_meta.crs),
                            "shape": (src_meta.height, src_meta.width),
                            "transform": str(src_meta.transform),
                            "nodata": src_meta.nodata,
                            "count": src_meta.count,
                        }
                except Exception as raster_err:
                    raster_meta = {"open_error": str(raster_err)}

            failing_indices = _find_failing_geometry_indices(tif_path, gdf, max_report=10)
            failing_geometries = []
            for idx in failing_indices:
                geometry = gdf.at[idx, "geometry"] if idx in gdf.index else None
                try:
                    validity_reason = explain_validity(geometry) if geometry is not None else "missing"
                except Exception:
                    validity_reason = "unknown"
                failing_geometries.append(
                    {
                        "index": idx,
                        "geom_type": getattr(geometry, "geom_type", None),
                        "is_empty": bool(getattr(geometry, "is_empty", False)) if geometry is not None else None,
                        "is_valid": bool(getattr(geometry, "is_valid", False)) if geometry is not None else None,
                        "validity": validity_reason,
                    }
                )

            logger.exception(
                "Population extract failure: country=%s year=%s tif=%s exists=%s gdf_rows=%s gdf_crs=%s null_geom=%s empty_geom=%s invalid_geom=%s geom_types=%s raster_meta=%s failing_geometry_sample=%s",
                country_hint,
                year,
                tif_path,
                raster_exists,
                len(gdf),
                gdf.crs,
                null_geom_count,
                empty_geom_count,
                invalid_geom_count,
                geom_type_counts,
                raster_meta,
                failing_geometries,
            )
            raise RuntimeError(
                f"Failed population extraction for country={country_hint}, year={year}, tif={tif_path}: {err}"
            ) from err

    # Restore the original CRS expected by downstream writers.
    if gdf.crs != org_crs:
        gdf = gdf.to_crs(org_crs)

    return gdf


def find_country_tif_files(countries, tif_dir):
    """Map ISO-2 codes to available country TIFF file lists.

    Parameters
    ----------
    countries : iterable[str]
        ISO-2 country codes to resolve.
    tif_dir : str
        Root directory containing per-country raster subdirectories named by ISO-3.

    Returns
    -------
    dict[str, list[str] | None]
        Mapping of ISO-2 code to TIFF paths or None when not found.
    """
    alpha_3_to_2, alpha_2_to_3, alpha_3_to_names, alpha_2_to_names = get_iso_codes()
    tif_filepaths = {}
    for iso_2 in countries:
        normalized_iso2 = _normalize_country_code(iso_2)
        if normalized_iso2 is None:
            continue

        if normalized_iso2 in alpha_2_to_3:
            iso_3 = alpha_2_to_3[normalized_iso2].lower()
            temp_dir = os.path.join(tif_dir, iso_3)
            if os.path.exists(temp_dir):
                tif_filepath = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".tif")]
                tif_filepaths[normalized_iso2] = tif_filepath if tif_filepath else None
            else:
                tif_filepaths[normalized_iso2] = None
        else:
            tif_filepaths[normalized_iso2] = None
    return tif_filepaths


def find_newest_country_tif_files(countries, tif_dir):
    """Map ISO-2 codes to their newest available country TIFF file.

    Parameters
    ----------
    countries : iterable[str]
        ISO-2 country codes to resolve.
    tif_dir : str
        Root directory containing per-country raster subdirectories named by ISO-3.

    Returns
    -------
    dict[str, str]
        Mapping of ISO-2 code to newest TIFF path.
    """
    tif_filepaths = find_country_tif_files(countries, tif_dir)
    newest = {}
    for country, files in tif_filepaths.items():
        if files is None:
            continue

        year_file = {}
        for file in files:
            if os.path.exists(file):
                tokens = os.path.basename(file).replace(".tif", "").split("_")
                parts = [int(token) for token in tokens if token.startswith("20") and token.isdigit()]
                if parts:
                    year_file[parts[0]] = file
                else:
                    logger.warning("Could not parse year from filename: %s", file)

        if year_file:
            newest[country] = year_file[max(year_file.keys())]
    return newest


def intersect_all_files(gdf, tif_dir, max_workers=16, all_years=True, country_col="ISO_2", years=None):
    """Intersect population rasters with polygons across all countries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons with a country-code column (see ``country_col``).
    tif_dir : str
        Root directory containing country raster subdirectories.
    max_workers : int, default=16
        Maximum number of worker processes.
    all_years : bool, default=True
        Whether to attach all discovered raster years instead of only the latest.
    country_col : str, default='ISO_2'
        Column on ``gdf`` containing ISO 3166-1 alpha-2 country codes.

    Returns
    -------
    geopandas.GeoDataFrame
        Concatenated result with population statistics attached.
    """
    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")
    if country_col not in gdf.columns:
        raise KeyError(f"country_col '{country_col}' not found in GeoDataFrame")

    gdf = gdf.copy()
    normalized = gdf[country_col].apply(_normalize_country_code)
    # Rows that cannot be matched to a raster are kept with NaN population
    # (not dropped) so the output has one row per input polygon and the
    # missing population is visible instead of silently absent.
    unmatched = [gdf[normalized.isna()]]
    if len(unmatched[0]):
        logger.warning(
            "%s rows with empty/invalid %s kept with NaN population",
            len(unmatched[0]),
            country_col,
        )
    gdf = gdf[normalized.notna()].copy()
    gdf[country_col] = normalized[normalized.notna()]

    requested_countries = sorted(gdf[country_col].unique().tolist())
    tif_filepaths = find_country_tif_files(requested_countries, tif_dir) if requested_countries else {}

    data = []
    countries = [c for c in requested_countries if c in tif_filepaths and tif_filepaths[c] is not None]
    missing_rasters = sorted(set(requested_countries) - set(countries))
    if missing_rasters:
        unmatched.append(gdf[gdf[country_col].isin(missing_rasters)])
        logger.warning(
            "%s/%s countries have no raster TIFFs; their %s rows are kept with NaN population (examples: %s)",
            len(missing_rasters),
            len(requested_countries),
            len(unmatched[-1]),
            ", ".join(missing_rasters[:20]),
        )

    logger.info(
        "Intersecting %s countries out of %s requested countries",
        len(countries),
        len(requested_countries),
    )

    random.shuffle(countries)
    failures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                intersect_single_file,
                gdf[gdf[country_col] == iso_2].copy(),
                tif_filepaths.get(iso_2, []),
                all_years=all_years,
                years=years,
            ): iso_2
            for iso_2 in countries
        }

        # Progress bar for country processing
        for future in tqdm(as_completed(futures), total=len(futures), desc="Intersecting countries"):
            if future is not None:
                try:
                    sub_gdf = future.result()
                    data.append(sub_gdf)
                except Exception as err:
                    iso_2 = futures.get(future, "UNKNOWN")
                    failures.append((iso_2, str(err)))
                    logger.error("Country intersection failed for %s: %s", iso_2, err, exc_info=True)
    if failures:
        preview = "; ".join([f"{iso}: {msg}" for iso, msg in failures[:10]])
        raise RuntimeError(f"Population raster intersection failed for {len(failures)} countries. Examples: {preview}")
    if not data:
        logger.warning("No data returned from any country - check raster files and polygon-raster intersection")
    parts = [part for part in data + unmatched if len(part)]
    if not parts:
        return gdf.iloc[0:0].copy()
    data = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(data, geometry="geometry", crs=gdf.crs)


def orchestrate_intersections(
    voronoi_file, tif_dir, output_path, max_workers=16, country_col="ISO_2", overwrite=True, years=None
):
    """Add population columns to one Voronoi layer and write it to ``output_path``.

    Parameters
    ----------
    voronoi_file : str
        Service-area GeoPackage of one approach (``create_output_paths(cfg)['voronoi'][approach]``).
    tif_dir : str
        Root directory containing country population rasters.
    output_path : str
        Population-enriched GeoPackage (``create_pop_output_paths(cfg)['voronoi'][approach]``).
    max_workers : int, default=16
        Maximum number of worker processes for zonal statistics.
    country_col : str, default='ISO_2'
        Column on the loaded GeoDataFrame containing ISO alpha-2 country codes.
    overwrite : bool, default=True
        Whether to redo the output if it already exists.
    years : set[int] | None
        Raster years to read (``population_years(cfg)``).
    """
    if os.path.exists(output_path) and not overwrite:
        logger.info(f"Output already exists at {output_path} and overwrite is False. Skipping.")
        return
    if not os.path.exists(voronoi_file):
        raise FileNotFoundError(f"Service areas not found (run create_voronoi first): {voronoi_file}")
    logger.info(f"Processing {os.path.basename(voronoi_file)}")

    gdf = gpd.read_file(voronoi_file)
    logger.info(f"Loaded Voronoi layer with {len(gdf)} features")
    gdf = intersect_all_files(gdf, tif_dir, max_workers, all_years=True, country_col=country_col, years=years)
    with atomic_output(output_path) as tmp_path:
        gdf.to_file(tmp_path, driver="GPKG", index=False)
    logger.info(f"Successfully saved population-enhanced file to {output_path}")


def main():
    """Add population to the service areas of one approach (``--approach 1``) of the configured variant."""
    parser = argparse.ArgumentParser(
        description="Intersect Voronoi polygons with population rasters and export enriched layers."
    )
    parser.add_argument(
        "--approach",
        required=True,
        choices=APPROACHES,
        help="Service-area approach whose layer gets population columns",
    )
    add_standard_override_arguments(parser)
    args = parser.parse_args()

    try:
        overrides = parse_config_overrides(args=args)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    cfg = load_config(script_name="add_pop", **overrides)
    provenance.write_run_manifest(cfg, "add_pop")
    voronoi_file = create_output_paths(cfg)["voronoi"][args.approach]
    output_path = create_pop_output_paths(cfg)["voronoi"][args.approach]
    logger.info(
        f"Configuration loaded: input={voronoi_file}, pop_tif_dir={cfg['paths']['pop_tif_dir']}, "
        f"max_workers={cfg['add_pop_max_workers']}"
    )
    try:
        orchestrate_intersections(
            voronoi_file,
            cfg["paths"]["pop_tif_dir"],
            output_path,
            cfg["add_pop_max_workers"],
            country_col=cfg["country_output_column"],
            overwrite=cfg["overwrite_existing"],
            years=population_years(cfg),
        )
        logger.info("Population data integration completed successfully")
    except Exception as err:
        logger.error(f"Population data integration failed: {err}", exc_info=True)
        sys.exit(1)


def population_years(cfg):
    """Raster years add_pop reads: the years download_pop fetches.

    The 2014 layer is the 1 km unconstrained product; it counts only with
    ``include_2014`` (a 2014 file left in pop_tif_dir from an older download is
    otherwise skipped instead of becoming a 2014_zonal_sum column).
    """
    years = set(range(int(cfg["start_year"]), int(cfg["end_year"]) + 1))
    if cfg["include_2014"]:
        years.add(2014)
    return years


if __name__ == "__main__":
    configure_logging()
    main()
