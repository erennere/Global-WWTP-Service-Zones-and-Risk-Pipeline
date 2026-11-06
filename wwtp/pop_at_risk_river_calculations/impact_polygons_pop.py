"""Generate environmental impact polygons from non-served population points.

The pipeline maps population loads to river segments, propagates decayed load
downstream by basin hierarchy, builds plume polygons segment-by-segment, and
merges outputs into a final GeoPackage.
"""

import argparse
import logging
import traceback
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from shapely.geometry import Polygon
from ..starter import add_standard_override_arguments, get_runtime_params, load_config, parse_config_overrides
from .. import provenance
from ..utils import atomic_output, configure_logging
from ..geo_utils import batch_estimate_utm_epsg

logger = logging.getLogger(__name__)


def impact_polygons_output_path(base_path, radius):
    """Canonical path of one radius's impact-polygon GeoPackage."""
    return base_path.replace(".gpkg", f"_{int(float(radius))}.gpkg")


# --- Global Placeholders (Populated by initializer in workers) ---
next_dict = {}
geom_dict = {}
lat_dict = {}
level_dict = {}
discharge_dict = {}

def create_dicts(river_gdf, next_col, id_col, main_riv_col, discharge_col, weight_col='weight'):
    """Build global lookup dictionaries used by worker processes.

    Parameters
    ----------
    river_gdf : geopandas.GeoDataFrame
        River network with topology, geometry, and discharge attributes.
    next_col : str
        Column containing downstream segment IDs.
    id_col : str
        Column containing current segment IDs.
    main_riv_col : str
        Column identifying the basin main stem.
    discharge_col : str
        Column containing discharge values.
    weight_col : str, default='weight'
        Column containing propagated branch weights.

    Notes
    -----
    Missing weights are converted to ``0.0`` before topology dictionaries are
    built so downstream propagation never receives ``NaN`` branch weights.
    """
    global next_dict, geom_dict, lat_dict, level_dict, discharge_dict
    logger.info("Building topology and hierarchical levels")

    # 1. Clean IDs and fill missing weights with 0
    # We keep the rows even if weight is NaN, but we drop rows with no IDs
    river_gdf = river_gdf.dropna(subset=[id_col, next_col, main_riv_col]).copy()
    
    river_gdf[id_col] = river_gdf[id_col].astype(int)
    river_gdf[next_col] = river_gdf[next_col].astype(int)
    river_gdf[main_riv_col] = river_gdf[main_riv_col].astype(int)
    
    river_gdf[weight_col] = river_gdf[weight_col].fillna(0.0).astype(float)

    # 2. next_dict: {id: (next_id, weight)}
    # Now getattr will always return a float (0.0 or higher)
    next_dict = {
        int(getattr(row, id_col)): (int(getattr(row, next_col)), float(getattr(row, weight_col))) 
        for row in river_gdf[[id_col, next_col, weight_col]].itertuples(index=False)
    }

    # 3. lat_dict
    lat_dict = dict(zip(river_gdf[id_col], river_gdf['lat'].astype(float)))

    # 4. geom_dict
    geom_dict = {}
    for utm_zone in river_gdf.utm.unique():
        if pd.isna(utm_zone): continue
        mask = river_gdf['utm'] == utm_zone
        epsg_val = int(float(utm_zone))
        sub_gdf = river_gdf[mask].copy().to_crs(epsg_val)
        #sub_gdf = river_gdf[mask].copy().to_crs(3857)
        geom_dict.update(dict(zip(sub_gdf[id_col], sub_gdf['geometry'])))

    # 5. level_dict (Topology building)
    upstream_adj = {}
    for rid, (nxt, _) in next_dict.items():
        if nxt != 0:
            upstream_adj.setdefault(nxt, []).append(rid)

    level_dict = {}
    unique_basins = river_gdf[main_riv_col].unique()
    for main_riv in tqdm(unique_basins, desc="Building Hierarchical Levels"):
        m_id = int(main_riv)
        levels, curr, visited = [], [m_id], {m_id}
        while curr:
            levels.append(curr)
            nxt_lvl = []
            for node in curr:
                for trib in upstream_adj.get(node, []):
                    if trib not in visited:
                        visited.add(trib)
                        nxt_lvl.append(trib)
            curr = nxt_lvl if nxt_lvl else None
        level_dict[m_id] = levels

    logger.info("Successfully processed %s basins", len(level_dict))

    #discharge info dict
    discharge_dict = dict(zip(river_gdf[id_col], river_gdf[discharge_col]))

def init_worker(shared_next, shared_geom, shared_lat, shared_level, shared_dis):
    """Initializes global dictionaries in each worker process once."""
    global next_dict, geom_dict, lat_dict, level_dict, discharge_dict
    next_dict = shared_next
    geom_dict = shared_geom
    lat_dict = shared_lat
    level_dict = shared_level
    discharge_dict = shared_dis

def calculate_load_ratio(
    pop,
    dis_av_cms,
    org_per_pop=60.0,
    c_limit=5.0,
    least_discharge_cms=0.269,
    load=None
):
    """Convert population or load inputs into a normalized concentration ratio.

    Parameters
    ----------
    pop : float | pandas.Series | numpy.ndarray | None
        Population-equivalent load source used when ``load`` is not provided.
    dis_av_cms : float | pandas.Series | numpy.ndarray | None
        River discharge in cubic meters per second.
    org_per_pop : float, default=60.0
        Organic load per population equivalent in grams per day.
    c_limit : float, default=5.0
        Concentration limit used to normalize the resulting load.
    least_discharge_cms : float, default=0.269
        Minimum discharge substituted when the supplied discharge is zero or
        missing.
    load : float | pandas.Series | numpy.ndarray | None, default=None
        Optional precomputed load value. When omitted, load is derived from
        ``pop`` and ``org_per_pop``.

    Returns
    -------
    float | pandas.Series | numpy.ndarray
        Load ratio normalized by ``c_limit``.

    Notes
    -----
    The function supports both vectorized array-like inputs and scalar inputs.
    Discharge is converted from cubic meters per second to liters per second
    before the concentration ratio is computed.
    """
    # convert g/day → mg/s
    org_per_pop = org_per_pop / 86.4

    # handle Series input (vectorized mode)
    if isinstance(dis_av_cms, (pd.Series, np.ndarray)):
        dis = dis_av_cms.copy()
        if isinstance(dis, pd.Series):
            dis = dis.fillna(0)
            dis = dis.where(dis > 0, least_discharge_cms)
        else:
            dis = dis.astype(float)
            dis[~(dis > 0)] = least_discharge_cms
        dis *= 1000  # m³/s → l/s
        if load is None:
            load = pop * org_per_pop / dis
        return load / c_limit

    # handle scalar input (single-basin downstream case); missing, zero and
    # negative discharge get the floor, as in the vectorized branch
    else:
        dis_av_cms = _effective_discharge(dis_av_cms, least_discharge_cms)
        dis_av_cms *= 1000  # m³/s → l/s
        if load is None:
            load = pop * org_per_pop / dis_av_cms if pop is not None else 0
        return load / c_limit

def invert_calculate_load(load_ratio, c_limit=5.0):
    """Convert a normalized load ratio back into an absolute concentration load."""
    return load_ratio * c_limit

def calculate_radius(load_ratio, impact_radius=1000):
    """Return the configured plume radius only when the load ratio exceeds one."""
    return impact_radius if load_ratio >=1 else 0.0

def calculate_kt(lat, base_k=0.23, theta=1.047, water_temp_equator_c=28.0):
    """Compute temperature-adjusted decay coefficient by latitude."""
    temp = water_temp_equator_c * np.cos(np.radians(abs(lat)))
    return base_k * (theta**(temp - 20))


def _effective_discharge(dis_av_cms, least_discharge_cms):
    """Discharge in m3/s with zero/missing replaced by the configured floor."""
    if dis_av_cms is None or not np.isfinite(dis_av_cms) or dis_av_cms <= 0:
        return float(least_discharge_cms)
    return float(dis_av_cms)


def mix_downstream_concentration(downstream_conc, upstream_exit_conc, upstream_dis_cms, downstream_dis_cms,
                                 least_discharge_cms=0.269):
    """Concentration in the downstream segment after an upstream segment joins it.

    Mass balance: the upstream flux ``C_up * Q_up`` is diluted into the
    downstream discharge ``Q_down`` (which, in HydroRIVERS, already contains
    ``Q_up``), so ``C_down = C_down_local + C_up * Q_up / Q_down``. Calling this
    once per tributary gives ``(C1*Q1 + C2*Q2 + L_local) / Q_down``. The ratio
    is capped at 1 so inconsistent discharge data can never concentrate a
    plume. Concentrations here are absolute (mg/l), not ratios.
    """
    q_up = _effective_discharge(upstream_dis_cms, least_discharge_cms)
    q_down = _effective_discharge(downstream_dis_cms, least_discharge_cms)
    return downstream_conc + upstream_exit_conc * min(1.0, q_up / q_down)


def generate_single_segment_plume(rid, lat, start_load_ratio=None, step_m=100.0, c_limit=5.0, base_k=0.23, theta=1.047,
                                  impact_radii=(1000, 2000), next_state_dict=None, velocity_m_s=1.0,
                                  water_temp_equator_c=28.0):
    """Generate plume polygons for one river segment and return the exit load.

    Parameters
    ----------
    rid : int
        River segment identifier.
    lat : float
        Segment latitude used for decay-rate estimation.
    start_load_ratio : float | None, default=None
        Incoming normalized load ratio for the segment.
    step_m : float, default=100.0
        Interpolation step in meters.
    c_limit : float, default=5.0
        Concentration limit used for normalization.
    base_k : float, default=0.23
        Base decay coefficient.
    theta : float, default=1.047
        Temperature adjustment factor.
    impact_radii : list[float], default=[1000, 2000]
        Plume half-widths to materialize around the segment.
    next_state_dict : dict | None, default=None
        Optional segment-state dictionary for this basin task. When omitted,
        the global topology dictionary is used.
    velocity_m_s : float, default=1.0
        Flow velocity that converts distance along the river into travel time.
    water_temp_equator_c : float, default=28.0
        Equatorial water temperature for the latitude temperature model.

    Returns
    -------
    tuple
        Tuple ``(polygons, exit_load)`` where ``polygons`` is a list of plume
        polygons or ``None`` when no plume survives the concentration threshold.
    """
    state_dict = next_dict if next_state_dict is None else next_state_dict

    if rid not in state_dict or rid not in geom_dict or rid not in discharge_dict:
        return None, 0.0
    
    if start_load_ratio is None:
        _, start_load_ratio = state_dict[rid]
    else:
        start_load_ratio = float(start_load_ratio)

    line = geom_dict[rid]
    seg_len = line.length
    kt = calculate_kt(lat, base_k=base_k, theta=theta, water_temp_equator_c=water_temp_equator_c)
    velocity_m_day = 86400.0 * float(velocity_m_s)

    # 1. Generate distances. Ensure at least two points for directionality.
    distances = np.arange(0, seg_len, step_m)
    if len(distances) < 2:
        distances = np.array([0.0, seg_len])
    
    times = distances / velocity_m_day
    load_ratios = start_load_ratio * np.exp(-kt * times)
    # load_ratios is already normalized by c_limit upstream, so threshold at 1.0 directly.
    mask = load_ratios >= 1.0

    # 2. Handle Truncation if plume dies mid-segment
    if not np.all(mask):
        stop_idx = np.where(~mask)[0][0]
        # If it dies at the very first point, we can't make a polygon
        if stop_idx < 2:
            return None, 0.0
        distances = distances[:stop_idx]
        load_ratios = load_ratios[:stop_idx]
        exit_load = 0.0
    else:
        # The sample points stop one step short of the segment end; the load
        # leaving the segment has decayed over its whole length.
        exit_ratio = start_load_ratio * np.exp(-kt * seg_len / velocity_m_day)
        exit_load = invert_calculate_load(exit_ratio, c_limit=c_limit)

    # 3. Vectorized Geometry Generation
    # np.atleast_2d ensures that even a single point is (1, 2) instead of (2,)
    points = np.array([line.interpolate(d).coords[0] for d in distances])
    points = np.atleast_2d(points)
    
    eps = 0.1 # Reduced epsilon for better precision on short segments
    lookahead = np.minimum(distances + eps, seg_len)
    next_pts = np.array([line.interpolate(d).coords[0] for d in lookahead])
    next_pts = np.atleast_2d(next_pts)
    
    diff = next_pts - points
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    
    # Avoid division by zero on zero-length segments
    norms[norms == 0] = 1.0
    tangents = diff / norms
    
    # Left/Right boundaries (Normal vectors)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    
    polygons = []
    for impact_radius in impact_radii:
        r_val = calculate_radius(start_load_ratio, impact_radius=impact_radius)
        
        # Calculate offset coordinates
        left_side = points + (normals * r_val)
        right_side = points - (normals * r_val)
        
        # Construct polygon ring: Left side forward, Right side backward
        coords = np.concatenate([left_side, right_side[::-1]], axis=0)
        
        try:
            poly = Polygon(coords)
            polygons.append(poly if poly.is_valid else poly.buffer(0))
        except Exception:
            continue
    return (polygons, exit_load) if polygons else (None, 0.0)
    
def create_impact_polygons(pop_chunk, main_riv, nxt_dis_col, model_params=None):
    """Generate impact polygons for one basin chunk.

    Parameters
    ----------
    pop_chunk : geopandas.GeoDataFrame
        Population features belonging to one basin main stem.
    main_riv : int
        Main-river identifier for the basin chunk.
    nxt_dis_col : str
        Column containing the downstream segment IDs used as plume targets.
    model_params : dict | None, default=None
        Runtime model parameters controlling decay and plume geometry.

    Returns
    -------
    dict
        Mapping from impact radius to a GeoDataFrame-like subset of plume
        geometries for the chunk.
    """
    global next_dict, discharge_dict
    try:
        if pop_chunk.empty: 
            return {}
        
        target_ids = set(pop_chunk[nxt_dis_col])

        # 1. Get the topological levels for this specific basin
        levels = level_dict.get(int(main_riv), [])
        if not levels:
            return {}

        c_limit = model_params['c_limit']
        base_k = model_params['base_k']
        theta = model_params['theta']
        step_m = model_params['step_m']
        least_discharge_cms = model_params['least_discharge_cms']
        impact_radii = model_params['impact_radii']
        velocity_m_s = model_params['velocity_m_s']
        water_temp_equator_c = model_params['water_temp_equator_c']

        # Isolate mutable load propagation state for this basin task only.
        basin_ids = {rid for level in levels for rid in level}
        local_next_dict = {
            rid: next_dict[rid]
            for rid in basin_ids
            if rid in next_dict
        }

        # result_geometries stores final polys for IDs in pop_chunk
        result_geometries = {}

        # 2. Process ALL levels from Upstream (N) to Downstream (0)
        for current_level_ids in reversed(levels):
            for rid in current_level_ids:
                state = local_next_dict.get(rid)
                if state is None:
                    continue

                down_id, load_ratio = state
                if load_ratio <= 0.0:
                    continue

                lat = lat_dict.get(rid, 0.0)
                polygons, exit_load = generate_single_segment_plume(
                    rid,
                    lat,
                    start_load_ratio=load_ratio,
                    step_m=step_m,
                    c_limit=c_limit,
                    base_k=base_k,
                    theta=theta,
                    impact_radii=impact_radii,
                    next_state_dict=local_next_dict,
                    velocity_m_s=velocity_m_s,
                    water_temp_equator_c=water_temp_equator_c,
                )

                # Hand over residual load to immediate downstream segment within this basin.
                downstream_state = local_next_dict.get(down_id)
                # The exit concentration is diluted by the downstream discharge
                # (flow-weighted mixing), not added on top of it.
                if down_id != 0 and downstream_state is not None and exit_load > 0.0:
                    next_next, downstream_load_ratio = downstream_state
                    downstream_conc = mix_downstream_concentration(
                        invert_calculate_load(downstream_load_ratio, c_limit),
                        exit_load,
                        discharge_dict.get(rid),
                        discharge_dict.get(down_id),
                        least_discharge_cms=least_discharge_cms,
                    )
                    local_next_dict[down_id] = (next_next, downstream_conc / c_limit)

                # 3. Save geometry if this segment is one of our target population points

                if rid in target_ids:
                    result_geometries[rid] = polygons

        # 4. Map the calculated geometries back to the pop_chunk
        results = {}
        for index, radius in enumerate(impact_radii):
            pc = pop_chunk.copy() 
            pc['geometry'] = pc[nxt_dis_col].map(result_geometries).apply(
                lambda geoms, i=index: geoms[i] if geoms is not None and len(geoms) > i else None)
            valid_returns = pc.dropna(subset=['geometry'])
            results[radius] = valid_returns[['geometry', 'utm', 'country', 'MAIN_RIV', nxt_dis_col]]
        return results

    except Exception as e:
        logger.exception("Error processing basin %s: %s", main_riv, e)
        logger.error(traceback.format_exc())
        return {}

def parallel_dissolve(subset_df, crs_code):
    """Worker function for dissolving a specific group of polygons."""
    if subset_df.empty:
        return gpd.GeoDataFrame()
    
    # Create GDF and perform spatial merge
    gdf = gpd.GeoDataFrame(subset_df, geometry='geometry', crs=int(crs_code))
    # dissolve() merges, explode() separates non-contiguous parts
    dissolved = gdf.dissolve().explode(index_parts=False)
    
    # Project to WGS84 here so the main process only has to concat
    return dissolved.to_crs(4326)

def orchestrate_logic(pop_gdf, nxt_dis_col, main_riv_col, max_workers, model_params=None):
    """Run plume generation, dissolve regional outputs, and assemble final layers.

    Parameters
    ----------
    pop_gdf : geopandas.GeoDataFrame
        Input population features with basin and downstream segment metadata.
    nxt_dis_col : str
        Column containing downstream segment IDs.
    main_riv_col : str
        Column containing main-river identifiers used for chunking.
    max_workers : int
        Maximum number of worker processes.
    model_params : dict | None, default=None
        Runtime model parameters controlling decay and plume geometry.

    Returns
    -------
    dict | None
        Mapping from impact radius to final GeoDataFrames, or ``None`` when no
        polygons are generated.
    """
    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")

    total_chunks = pop_gdf[main_riv_col].unique()
    results_list = {}
    
    # --- STAGE 1: Parallel Plume Generation ---
    logger.info("Distributing %s chunks to %s workers", len(total_chunks), max_workers)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(next_dict, geom_dict, lat_dict, level_dict, discharge_dict)
    ) as executor:
        
        futures = [
            executor.submit(
                create_impact_polygons,
                pop_gdf[pop_gdf[main_riv_col] == main_riv].copy(),
                main_riv,
                nxt_dis_col,
                model_params,
            ) for main_riv in total_chunks
        ]
        
        for future in tqdm(as_completed(futures), total=len(total_chunks), desc="Generating Plumes"):
            try:
                res = future.result()
                if res is not None and res:
                    for radius, df in res.items():
                        if radius not in results_list:
                            results_list[radius] = []
                        if not df.empty:
                            results_list[radius].append(df)
            except Exception as e:
                logger.exception("Plume generation failed: %s", e)

    if not results_list:
        logger.error("No valid polygons were generated")
        return None
    
    # --- STAGE 2: Parallel Regional Dissolve ---
    gdfs = {}
    for radius, dfs in results_list.items():
        final_df = pd.concat(dfs, ignore_index=True)
        unique_utms = final_df['utm'].unique()
        projected_results = []
        
        logger.info("Parallelizing dissolve across %s UTM groups", len(unique_utms))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit slices based on UTM group
            # Using 3857 for spatial operations to ensure global continuity
            dissolve_futures = {
                executor.submit(parallel_dissolve, final_df[final_df['utm'] == utm].copy(), utm): utm  
                for utm in unique_utms
            }
            
            for future in tqdm(as_completed(dissolve_futures), total=len(dissolve_futures), desc="Parallel Dissolving"):
                try:
                    res = future.result()
                    if res is not None and not res.empty:
                        projected_results.append(res)
                except Exception as e:
                    logger.exception("Dissolve error for UTM group %s: %s", dissolve_futures[future], e)

        # --- STAGE 3: Final Global Cleanup ---
        if not projected_results:
            continue

        logger.info("Merging regional results and performing final global cleanup")
        combined_gdf = gpd.GeoDataFrame(pd.concat(projected_results, ignore_index=True), crs=4326)

        try:
            # 1. Fix any potential invalidities before dissolving
            combined_gdf['geometry'] = combined_gdf['geometry'].make_valid()
            
            # 2. Use a tiny buffer to merge near-identical points (snapping)
            # This solves 'side location conflict' by giving GEOS a tiny bit of breathing room
            combined_gdf['geometry'] = combined_gdf['geometry'].buffer(1e-9) 
            
            # 3. Perform the dissolve
            dissolved = combined_gdf.dissolve()
            
            # 4. Clean up the resulting geometries
            gdfs[radius] = dissolved.explode(index_parts=False).make_valid()
            
        except Exception as e:
            logger.warning("Standard dissolve failed: %s. Attempting robust union", e)
            # Fallback: Union everything at once which is sometimes more stable than dissolve
            all_geoms = combined_gdf.geometry.values
            merged = unary_union(all_geoms)
            
            final_gdf = gpd.GeoDataFrame(geometry=[merged], crs=4326).explode(index_parts=False)
            gdfs[radius] = final_gdf.make_valid()

    if not gdfs:
        logger.error("No valid polygons were generated after global cleanup")
        return None
    return gdfs

def main():
    """Load inputs, generate impact polygons, and write final output.

    CLI usage::

        python -m ...impact_polygons_pop --max-workers 64 [--level ... --version ...]

    ``weight_func`` accepts ``mult``, ``add``, or ``""`` for default.
    """
    parser = argparse.ArgumentParser(
        description="Generate downstream impact polygons from non-served population data."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=64,
        help="Maximum worker processes for plume generation and dissolve",
    )
    parser.add_argument(
        "--radius",
        default=None,
        help=(
            "Optional impact radius (must match one of "
            "impact_polygons_pop_params.impact_radii in config.yaml); process "
            "only this one radius instead of every configured radius. Unset "
            "processes the full configured list - unchanged from the default."
        ),
    )
    add_standard_override_arguments(parser)
    args = parser.parse_args()

    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name="impact_polygons_pop", **overrides)
    provenance.write_run_manifest(cfg, "impact_polygons_pop")
    model_params = get_runtime_params(cfg)

    if args.radius is not None:
        requested_radius = float(args.radius)
        configured_radii = model_params['impact_radii']
        if not any(requested_radius == radius for radius in configured_radii):
            raise ValueError(
                f"Unknown --radius '{args.radius}'; expected one of "
                f"{sorted(int(radius) for radius in configured_radii)} "
                "(impact_polygons_pop_params.impact_radii)."
            )
        model_params['impact_radii'] = [requested_radius]
        logger.info("Restricting to single --radius override: %s", requested_radius)

    logger.info("Using runtime model params: %s", model_params)

    # 1. Load Data
    logger.info("Loading datasets")
    
    # Only pull the ID and Population columns for the settlements
    pop_cols = ['NXT_DIS', 'pop_sum', 'country', 'utm']
    pop_gdf = gpd.read_file(
        cfg['paths']['non_served_nxt_river_outpath'], 
        columns=pop_cols, 
    ).to_crs("EPSG:4326")
    
    river_cols = ['HYRIV_ID', 'NEXT_DOWN', 'geometry', 'MAIN_RIV', 'DIS_AV_CMS']
    river_gdf = gpd.read_file(
        cfg['paths']['rivershed_output_path'],
        columns=river_cols
    ).to_crs("EPSG:4326")
    logger.info("Loaded %s population rows and %s river rows", len(pop_gdf), len(river_gdf))

    # 2. Vectorized Processing (Fast)
    pop_gdf = pop_gdf[pop_gdf['NXT_DIS'].notna() & pop_gdf['pop_sum'].notna()].reset_index(drop=True)
    pop_gdf['HYRIV_ID'] = pop_gdf['NXT_DIS'].astype(int)
    pop_gdf['pop_sum'] = pop_gdf['pop_sum'].astype(int)
    pop_gdf = pop_gdf[(pop_gdf['HYRIV_ID'] != -1) & (pop_gdf['pop_sum'] > 0)].copy()
    pop_gdf.drop(['NXT_DIS'], axis=1, inplace=True)

    river_gdf = river_gdf[river_gdf['NEXT_DOWN'].notna()]
    river_gdf['HYRIV_ID'] = river_gdf['HYRIV_ID'].astype(int)
    river_gdf['NEXT_DOWN'] = river_gdf['NEXT_DOWN'].astype(int)
    river_gdf['utm'], river_gdf['lat'] = batch_estimate_utm_epsg(river_gdf)

    pop_gdf = pop_gdf.merge(river_gdf[['HYRIV_ID', 'MAIN_RIV', 'DIS_AV_CMS']], on='HYRIV_ID', how='left')
    pop_gdf['env_load'] = calculate_load_ratio(pop_gdf['pop_sum'], pop_gdf['DIS_AV_CMS'], org_per_pop=model_params['org_per_pop'], c_limit=model_params['c_limit'],
                                          least_discharge_cms=model_params['least_discharge_cms'], load=None)
    pop_gdf = pop_gdf.dropna(subset=['MAIN_RIV', 'env_load']).reset_index(drop=True)
    pop_gdf.drop(['pop_sum'], axis=1, inplace=True)
    
    river_gdf = river_gdf.merge(pop_gdf[['HYRIV_ID', 'env_load']], on='HYRIV_ID', how='left')
    river_gdf = river_gdf[river_gdf['MAIN_RIV'].isin(pop_gdf['MAIN_RIV'].unique())].reset_index(drop=True)
    #pop_gdf = pop_gdf[pop_gdf['MAIN_RIV'].isin(pop_gdf['MAIN_RIV'].unique()[0:10])].reset_index(drop=True)
    logger.info("Prepared %s filtered population rows and %s filtered river rows", len(pop_gdf), len(river_gdf))

    # 3. Build State
    create_dicts(river_gdf, 'NEXT_DOWN', 'HYRIV_ID', 'MAIN_RIV', 'DIS_AV_CMS', 'env_load')
    
    # 4. Process
    max_workers = int(args.max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    results = orchestrate_logic(
        pop_gdf,
        'HYRIV_ID',
        'MAIN_RIV',
        max_workers=max_workers,
        model_params=model_params,
    )
    
    if results is not None:
        for radius, gdf in results.items():
            output_path = impact_polygons_output_path(cfg['paths']['impact_pop_polygons_outpath'], radius)
            with atomic_output(output_path) as tmp_path:
                gdf.to_file(tmp_path, driver='GPKG')
            logger.info("Process complete. Wrote %s geometries for radius %s", len(gdf), radius)
    else:
        logger.warning("No output generated")

if __name__ == "__main__":
    configure_logging()
    main()