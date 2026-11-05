"""
Population Data Processing Module

Downloads and processes global population data from WorldPop.
Converts point-based population data (CSV) to raster format (GeoTIFF)
and creates country-level mosaics.

Workflow:
1. Retrieve download URLs for all countries from WorldPop
2. Download population data (rasters or point clouds)
3. For CSVs: rasterize point data to GeoTIFF with UTM projection
4. Mosaic multiple tiles into single country-level GeoTIFF files

Output directories:
- ../data/population/zipped/      : Downloaded archives
- ../data/population/unzipped/    : Extracted files
- ../data/population/rasterized/  : Individual CSV rasterized outputs
- ../data/population/merged/      : Country mosaics, one per year (pop_<iso3>_<year>_merged.tif)
"""
import argparse
import requests
import zipfile
import os, re, shutil
from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from rasterio.transform import from_origin
import numpy as np
import rasterio
from rasterio.merge import merge
import logging
from .starter import add_standard_override_arguments, load_config, parse_config_overrides
from . import provenance
from .utils import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    configure_logging,
    ensure_output_dir_for_file,
    get_iso_codes,
    requests_session_with_retries,
)

logger = logging.getLogger(__name__)

def extract_first_wildcard(test_string, pattern):
    """Extract first capture group from regex pattern match.
    
    Args:
        test_string: String to search
        pattern: Regex pattern with one capture group
        
    Returns:
        First capture group content or None if no match
    """
    match = re.search(pattern, test_string)
    if match:
        return match.group(1)
    return None

def try_extract_country(name, patterns):
    """Try multiple regex patterns against a filename and return first match."""
    for pattern in patterns:
        result = extract_first_wildcard(name, pattern)
        if result is not None:
            return result
    return None

def add_country_url(country_urls, country, url):
    """Add a URL to the country_urls dict, creating the list if needed."""
    if country in country_urls:
        country_urls[country].append(url)
    else:
        country_urls[country] = [url]

def get_urls_from_hdx():
    """Fetch population URLs from HDX for legacy workflows.

    This helper is not used by the default `main()` workflow, which currently
    builds WorldPop URLs directly in `get_urls()`. It is retained as a fallback
    for cases where HDX-hosted artifacts need to be revisited manually.
    """
    from hdx.utilities.easy_logging import setup_logging
    from hdx.api.configuration import Configuration
    from hdx.data.dataset import Dataset
    setup_logging()
    Configuration.create(hdx_site="prod", user_agent="HeiGIT", hdx_read_only=True)
    country_set = set()
    country_urls = {}
    datasets = Dataset.search_in_hdx("Meta Population")

    remaining_countries = {
        'ANR' : ['ANR_men_geotiff.zip', 'ANR_women_geotiff.zip'],
        'AUS' : ['population_aus_northeast_2018-10-01_geotiff.zip','population_aus_northwest_2018-10-01_geotiff.zip',
                'population_aus_southeast_2018-10-01_geotiff.zip','population_aus_southwest_2018-10-01_geotiff.zip'],
        'BRA' : [
            'population_bra_northeast_2018-10-01.geotiff.zip',
            'population_bra_northwest_2018-10-01.geotiff.zip',
            'population_bra_southeast_2018-10-01.geotiff.zip',
            'population_bra_southwest_2018-10-01.geotiff.zip'],
        'DOM' : ['population_dom_2018-10-01.geotiff.zip'],
        'FJI' : ['population_fji_2018-10-01_geotiff.zip'],
        'GIB' : ['gib_men_2020_geotiff.zip', 'gib_women_2020_geotiff.zip'],
        'HKG' : ['population_hkg_2018-10-01_geotiff.zip'],
        'HND' : ['population_hnd_2018-10-01.geotiff.zip'],
        'HTI' : ['population_hti_2018-10-01.geotiff.zip'],
        'IMN' : ['IMN_men_2019-08-03_csv.zip', 'IMN_women_2019-08-03_csv.zip'],
        'IND' : ['population_ind_pak_general.zip'],
        'IRL' : ['population_irl_2019-07-01_geotiff.zip'],
        'JAM' : ['population_jam_2018-10-01.geotiff.zip'],
        'KAZ' : ['population_kaz_2018-10-01_geotiff.zip'],
        'KIR' : ['population_kir_2018-10-01_geotiff.zip'],
        'KNA' : ['population_kna_2018-10-01.geotiff.zip'],
        'KOR' : ['population_kor_2018-10-01_geotiff.zip'],
        'LCA' : ['population_lca_2018-10-01.geotiff.zip'],
        'MEX' : ['population_mex_2018-10-01.geotiff.zip'],
        'MNG' : ['population_mng_2018-10-01_geotiff.zip'],
        'MSR' : ['population_msr_2018-10-01.geotiff.zip'],
        'NCL' : ['population_ncl_2018-10-01_geotiff.zip'],
        'NIC' : ['population_nic_2018-10-01.geotiff.zip'],
        'NPL' : ['population_npl_2018-10-01_geotiff.zip'],
        'NZL' : ['population_nzl_2018-10-01_geotiff.zip'],
        'PAK' : ['population_ind_pak_general.zip'],
        'PAN' : ['population_pan_2018-10-01.geotiff.zip'],
        'PRI' : ['population_pri_2018-10-01.geotiff.zip'],
        'PYF' : ['population_pyf_2018-10-01_geotiff.zip'],
        'REU' : ['population_reu_2018-10-01_geotiff.zip'],
        'SLB' : ['population_slb_2018-10-01_geotiff.zip'],
        'SLV' : ['population_slv_2018-10-01.geotiff.zip'],
        'SVN' : ['population_svn_2019-07-01_geotiff.zip'],
        'SYC' : ['population_syc_2018-10-01_geotiff.zip'],
        'TTO' : ['population_tto_2018-10-01.geotiff.zip'],
        'TUR' : ['population_turkey_2020_tif.zip'],
        'USA' : ['population_usa.part_1_of_6.csv.zip','population_usa.part_2_of_6.csv.zip',
                'population_usa.part_3_of_6.csv.zip','population_usa.part_4_of_6.csv.zip',
                'population_usa.part_5_of_6.csv.zip','population_usa.part_6_of_6.csv.zip'],
        'UZB' : ['population_uzb_2018-10-01_geotiff.zip'],
        'VCT' : ['population_vct_2018-10-01.geotiff.zip'],
        'VGB' : ['population_vgb_2018-10-01.geotiff.zip'],
        'VIR' : ['population_vir_2018-10-01.geotiff.zip'],
        'VUT' : ['population_vut_2018-10-01_geotiff.zip'],
        'WLF' : ['population_wlf_2018-10-01_geotiff.zip'],
        'WSM' : ['population_wsm_2018-10-01_geotiff.zip']
    }

    patterns = [
        r'^population_([a-z]{3})\.geotiff\.zip$',
        r'^(.*?)_general_(.*?)_geotiff\.zip$',
        r'^([a-z]{3})_population_\d{4}_geotiff\.zip$',
        r'^([a-z]{3})_general_.*_csv\.zip$',
        r'^([a-z]{3})_population_\d{4}_csv\.zip$'
    ]
    all_vals_flattened = [v for key, val in remaining_countries.items() for v in val]
    all_vals_flattened_keys  = [key for key, val in remaining_countries.items() for v in val]
    all_vals_dict = {k:v.lower() for k, v in zip(all_vals_flattened, all_vals_flattened_keys)}

    for i, elm in enumerate(datasets):
        for j, res in enumerate(elm.get_resources()):
            url = res.get('download_url')
            name = res.get('name')
            
            # Try to extract country code from filename patterns
            country = try_extract_country(name, patterns)
            
            if country is not None:
                if country in country_set:
                    continue
                country_set.add(country)
                add_country_url(country_urls, country, url)
            elif name in all_vals_flattened:
                iso3 = all_vals_dict[name]
                if iso3 in country_urls:
                    continue
                country_set.add(iso3)
                add_country_url(country_urls, iso3, url)
    return country_urls

def get_urls(start_year=2015, end_year=2024, worldpop_2014_template=None, worldpop_yearly_template=None,
             include_2014=False):
    """
    Generate WorldPop population data URLs for all countries.
    Returns a dictionary mapping country codes to lists of download URLs.

    The 2014 layer is the 1 km unconstrained product, not the 100 m
    constrained R2024B series used for 2015-2024, so it is only requested
    when ``include_2014`` is true.
    """
    alpha_3_to_2, alpha_2_to_3, alpha_3_to_names, alpha_2_to_names = get_iso_codes()
    all_countries = list(alpha_3_to_2.keys())
    if worldpop_2014_template is None:
        worldpop_2014_template = 'https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2014/{country_upper}/{country_lower}_ppp_2014_1km_Aggregated.tif'
    if worldpop_yearly_template is None:
        worldpop_yearly_template = 'https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/{year}/{country_upper}/v1/100m/constrained/{country_lower}_pop_{year}_CN_100m_R2024B_v1.tif'

    country_urls = {
        k.lower(): (
            [worldpop_2014_template.format(country_upper=k.upper(), country_lower=k.lower())] if include_2014 else []
        )
        for k in all_countries
    }
    for k in country_urls.keys():
        for year in range(int(start_year), int(end_year) + 1):
            country_urls[k].append(
                worldpop_yearly_template.format(
                    year=str(int(year)),
                    country_upper=k.upper(),
                    country_lower=k.lower(),
                )
            )
    return country_urls

def download_file(url, output_path):
    """Stream a remote file to disk using an adaptive chunk size."""
    try:
        session = requests_session_with_retries()
        with session.get(url, stream=True, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS) as response:
            response.raise_for_status()

            total_size = int(response.headers.get('Content-Length', 0))
            if total_size == 0:
                chunk_size = 8192
            elif total_size < 10 * 1024 * 1024:
                chunk_size = 8192
            elif total_size < 100 * 1024 * 1024:
                chunk_size = 65536
            else:
                chunk_size = 262144
            print(f"Downloading with chunk size: {chunk_size // 1024} KB")
            ensure_output_dir_for_file(output_path)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk: 
                        f.write(chunk)
        print(f"Saved to {output_path}")
        return True
    except requests.RequestException as e:
        print(f"Download error for {url}: {e}")
        return False

def download_save_and_unzip_pop(url, country, data_dir='../data/population'):
    """Download and extract population data file.
    
    Args:
        url: Download URL for the file
        country: ISO3 country code
        data_dir: Base directory for data storage
        
    Returns:
        Path to extracted folder or None if download/extraction failed
    """
    filename = url.split('/')[-1]
    zip_folder = os.path.join(data_dir, 'zipped')
    extract_folder = os.path.join(data_dir, 'unzipped', country)
    zip_filename = os.path.join(zip_folder, filename)

    os.makedirs(zip_folder, exist_ok=True)
    os.makedirs(extract_folder, exist_ok=True)

    if url.endswith('zip'):
        try:
            response = requests.get(url)
            ensure_output_dir_for_file(zip_filename)
            with open(zip_filename, "wb") as f:
                f.write(response.content)
        except Exception as err:
            logger.warning(f'Download failed for {url}: {err}')
            try:
                logger.info('Attempting fallback URL with maxar_v1')
                response = requests.get(url.replace('BSGM', 'maxar_v1'))
                ensure_output_dir_for_file(zip_filename)
                with open(zip_filename, "wb") as f:
                    f.write(response.content)
            except Exception as err:
                logger.error(f'Fallback download also failed: {err}')
                return None
        try:
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
        except Exception as err:
            logger.error(f'Extraction failed for {zip_filename}: {err}') 
            return None
        return extract_folder
    else:
        output_file = os.path.join(extract_folder, os.path.basename(url))
        if not download_file(url, output_file):
            logger.info('Attempting fallback URL with maxar_v1')
            if not download_file(url.replace('BSGM', 'maxar_v1'), output_file):
                return None
        return extract_folder
            
def download_save_and_unzip_pops(country_urls, country, data_dir='../data/population'):
    """Download and extract all population files for a country.
    
    Args:
        country_urls: Dictionary mapping countries to lists of URLs
        country: ISO3 country code
        data_dir: Base directory for data storage
        
    Returns:
        Path to extracted folder or None if all downloads failed
    """
    urls = country_urls[country]
    extract_folder = None
    for url in urls:
        result = download_save_and_unzip_pop(url, country, data_dir)
        if result is not None:
            extract_folder = result
    return extract_folder

def find_type(folder, file_type):
    """Recursively find all files with given extension in folder.
    
    Args:
        folder: Directory path
        file_type: File extension to search for (e.g., '.tif')
        
    Returns:
        List of file paths matching the extension
    """
    result = []
    for entry in os.scandir(folder):
        if entry.is_file() and entry.name.endswith(file_type):
            result.append(entry.path)
        elif entry.is_dir():
            result.extend(find_type(entry.path, file_type))
    return result

def find_files(folder):
    """Find GeoTIFF or CSV files in folder hierarchy.
    
    Args:
        folder: Directory path
        
    Returns:
        Tuple of (file_list, is_tif_format)
        - file_list: List of found .tif or .csv files
        - is_tif_format: True if TIFFs found, False if CSVs
    """
    result = find_type(folder, '.tif')
    if not result:
        return find_type(folder, '.csv'), False
    return result, True
    
METRES_PER_DEGREE_LAT = 111_320.0


def rasterize_csv(df, output_path, res=30):
    """Rasterize a point-based population CSV into an EPSG:4326 GeoTIFF of summed counts.

    Each point adds its population to the cell it falls in, on a grid of
    ``res`` metres of latitude per cell (degrees, so no reprojection follows).
    The raster total equals the CSV total exactly. The previous version burned
    points into a UTM grid (a later point in the same cell replaced the
    earlier one), cut off the last row and column, and reprojected the counts
    to EPSG:4326 with nearest-neighbour resampling, which duplicates or drops
    counts.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing latitude, longitude, and population columns.
    output_path : str
        Path of the GeoTIFF to create.
    res : int | float, default=30
        Cell size in metres of latitude (``res / 111,320`` degrees).

    Returns
    -------
    str | None
        Output path when rasterization succeeds, otherwise ``None``.
    """
    df.columns = [c.lower() for c in df.columns]

    try:
        lat_col = next(c for c in df.columns if c.startswith('lat'))
        lon_col = next(c for c in df.columns if c.startswith('lon'))
        pop_col = next(c for c in df.columns if 'pop' in c or 'general' in c)
    except StopIteration:
        logger.error(f"Missing required column in CSV. Available columns: {list(df.columns)}")
        return None

    lat = pd.to_numeric(df[lat_col], errors='coerce').to_numpy(dtype=float)
    lon = pd.to_numeric(df[lon_col], errors='coerce').to_numpy(dtype=float)
    pop = pd.to_numeric(df[pop_col], errors='coerce').to_numpy(dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(pop)
    if not ok.any():
        logger.error("No valid points to rasterize for %s", output_path)
        return None
    lat, lon, pop = lat[ok], lon[ok], pop[ok]

    cell = float(res) / METRES_PER_DEGREE_LAT
    west, north = lon.min() - cell / 2, lat.max() + cell / 2  # points sit inside their cells, not on edges
    cols = np.floor((lon - west) / cell).astype(np.int64)
    rows = np.floor((north - lat) / cell).astype(np.int64)
    grid = np.zeros((rows.max() + 1, cols.max() + 1), dtype=np.float64)
    np.add.at(grid, (rows, cols), pop)

    ensure_output_dir_for_file(output_path)
    with rasterio.open(output_path, "w", driver='GTiff', height=grid.shape[0], width=grid.shape[1], count=1,
                       dtype='float32', crs="EPSG:4326", transform=from_origin(west, north, cell, cell),
                       compress='lzw') as dst:
        dst.write(grid.astype('float32'), 1)
    return output_path

def mosaic_large_rasters(raster_files, output_path):
    """Mosaic raster tiles of ONE year into a single output file.

    Uses ``rasterio.merge`` so nodata cells never enter the counts and an
    overlap is taken from the first tile instead of being summed. Tiles with a
    different resolution or CRS are rejected: population counts cannot be
    resampled to another cell size without changing the totals.

    Parameters
    ----------
    raster_files : list[str]
        Raster files to combine (one year, one product).
    output_path : str
        Destination path of the merged raster.
    """
    ensure_output_dir_for_file(output_path)
    if len(raster_files) == 1:
        shutil.copy(raster_files[0], output_path)
        return

    sources = [rasterio.open(fp) for fp in raster_files]
    try:
        resolutions = {tuple(round(v, 12) for v in src.res) for src in sources}
        crss = {src.crs.to_string() if src.crs else None for src in sources}
        if len(resolutions) > 1 or len(crss) > 1:
            raise ValueError(
                f"Cannot mosaic {output_path}: tiles differ in resolution {sorted(resolutions)} "
                f"or CRS {sorted(map(str, crss))}"
            )
        nodata = next((src.nodata for src in sources if src.nodata is not None), None)
        data, transform = merge(sources, nodata=nodata, method='first')
        profile = sources[0].profile.copy()
    finally:
        for src in sources:
            src.close()

    profile.update(
        driver='GTiff', height=data.shape[1], width=data.shape[2], count=data.shape[0],
        transform=transform, nodata=nodata, compress='lzw',
    )
    profile.pop('blockxsize', None)
    profile.pop('blockysize', None)
    profile.pop('tiled', None)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)


_YEAR_TOKEN = re.compile(r'(?<!\d)(20\d{2})(?!\d)')


def group_rasters_by_year(raster_files):
    """Group raster paths by the four-digit year in their file name.

    Different years must never be mosaicked together; files without a year
    are grouped under ``None``.
    """
    groups = {}
    for fp in raster_files:
        match = _YEAR_TOKEN.search(os.path.basename(fp))
        groups.setdefault(int(match.group(1)) if match else None, []).append(fp)
    return groups


def merged_output_path(pop_dir, country, year=None):
    """Canonical path of one country's merged population mosaic for one year."""
    year_part = f'_{int(year)}' if year is not None else ''
    return os.path.join(pop_dir, 'merged', f'pop_{country.lower()}{year_part}_merged.tif')


def process_single_country(country_urls, country, res=30, data_dir='../data/population'):
    """Download, prepare, and mosaic population data for one country.

    Parameters
    ----------
    country_urls : dict[str, list[str]]
        Mapping from ISO-3 country codes to source URLs.
    country : str
        ISO-3 country code to process.
    res : int | float, default=30
        Rasterization resolution used for CSV sources.
    data_dir : str, default='../data/population'
        Base directory for downloaded and processed files.
    """
    extract_folder = download_save_and_unzip_pops(country_urls, country, data_dir)
    if extract_folder is None:
        logger.warning(f'Failed to download data for {country}')
        return None
    result, if_tif = find_files(extract_folder)

    merged_path = merged_output_path(data_dir, country)
    output_path = os.path.join(data_dir, 'rasterized', country)
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    if if_tif:
        # One mosaic per year: the downloads hold one raster per year, and
        # mosaicking them together summed 2014-2024 into a single layer.
        for year, files in sorted(group_rasters_by_year(result).items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
            mosaic_large_rasters(files, merged_output_path(data_dir, country, year))
    else:
        filepaths = []
        for index, csv_filepath in enumerate(result):
            part_output_path = os.path.join(output_path, f'pop_{country}_part_{int(index+1)}.tif')
            try:
                df = pd.read_csv(csv_filepath)
                result_path = rasterize_csv(df, part_output_path, res)
                if result_path is not None:
                    filepaths.append(part_output_path)
            except Exception as err:
                logger.error(f'Failed to rasterize {csv_filepath}: {err}')
                continue
        filepaths = [f for f in filepaths if os.path.exists(f)]
        if filepaths:
            mosaic_large_rasters(filepaths, merged_path)
        else:
            logger.error(f'No successfully rasterized files for {country}')

def process_all_countries(country_urls, res=30, max_workers=16, data_dir='../data/population'):
    """Process population data for all configured countries in parallel.

    Parameters
    ----------
    country_urls : dict[str, list[str]]
        Mapping from ISO-3 codes to source URLs.
    res : int | float, default=30
        Rasterization resolution used for CSV sources.
    max_workers : int, default=16
        Maximum number of worker processes.
    data_dir : str, default='../data/population'
        Base directory for downloaded and processed files.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_country, country_urls, country, res, data_dir) 
                   for country in country_urls.keys()]
        
        # Progress bar for country processing
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing countries"):
            try:
                future.result()
            except Exception as err:
                logger.error(f'Error processing country: {err}')

def parse_args():
    """Parse the standardized named config-override flags plus download_pop's
    own --country selector (script-local, not part of the shared 7-field
    override contract - no other script needs a country selector).
    """
    parser = argparse.ArgumentParser(description="Run download_pop.")
    add_standard_override_arguments(parser)
    parser.add_argument(
        "--country",
        default=None,
        help=(
            "Optional ISO-3 country code (case-insensitive); process only "
            "this one country instead of every country. Unset processes all "
            "countries (subject to config's country_limit) - unchanged from "
            "the default."
        ),
    )
    return parser.parse_args()


def main(res=30, max_workers=8):
    """Load config, collect download URLs, and run the population pipeline.

    Parameters
    ----------
    res : int | float, default=30
        Rasterization resolution used when the function is called directly.
    max_workers : int, default=8
        Worker count used when the function is called directly.

    Notes
    -----
    The CLI entrypoint accepts the seven standard named config-override flags
    (``--level``, ``--version``, ``--buffer``, ``--weight-method``,
    ``--weight-func``, ``--dynamic-buffering``, ``--size-measure``), plus
    its own ``--country``. Resolution and worker count remain direct
    Python-call parameters unless separate CLI support is added.
    """
    args = parse_args()
    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name="download_pop", **overrides)
    provenance.write_run_manifest(cfg, "download_pop")
    data_dir = os.path.abspath(cfg["paths"]["pop_dir"])

    logger.info('Retrieving population data URLs')
    country_urls = get_urls(
        start_year=cfg['start_year'],
        end_year=cfg['end_year'],
        worldpop_2014_template=cfg['worldpop_2014_url_template'],
        worldpop_yearly_template=cfg['worldpop_yearly_url_template'],
        include_2014=bool(cfg['include_2014']),
    )

    if args.country:
        country = args.country.strip().lower()
        if country not in country_urls:
            raise ValueError(
                f"Unknown --country '{args.country}'; expected an ISO-3 code "
                "such as 'usa' (see src.utils.get_iso_codes)."
            )
        country_urls = {country: country_urls[country]}
        logger.info("Restricting to single --country override: %s", country)
    else:
        # 0 means "no limit"; YAML cannot express a literal null here because
        # the config resolver reads null as "inherit from an earlier section".
        country_limit = int(cfg['country_limit'])
        if country_limit < 0:
            raise ValueError("country_limit must be >= 0 (0 disables the limit)")
        if country_limit:
            selected_countries = sorted(country_urls.keys())[:country_limit]
            country_urls = {k: country_urls[k] for k in selected_countries}
            logger.info("Applying configured country limit: %s", country_limit)

    logger.info(f'Processing {len(country_urls)} countries with {max_workers} workers')
    process_all_countries(country_urls, res, max_workers, data_dir)
    logger.info('Population data processing complete')

if __name__ == '__main__':
    configure_logging()
    main()