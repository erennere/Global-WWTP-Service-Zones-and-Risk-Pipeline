"""Download OSM line and polygon features for each annotation grid cell.

For every grid feature, the script queries Overpass, converts the response into
GeoDataFrames, and writes one GeoJSON per grid cell for lines and polygons.
"""

import argparse
import os, json, time, re
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..utils import atomic_output, configure_logging, requests_session_with_retries
from ..geo_utils import repair_geometry

# (key, value) pairs; a list, not a dict, because one key ("waterway") is
# queried with two values - a dict literal silently kept only the last one,
# so streams were never requested. An empty value means "any value".
queries = [
    ("man_made", "wastewater_plant"),
    ("waterway", "stream"),
    ("waterway", "river"),
    ("landuse", ""),
    ("power", "plant"),
    ("industrial", ""),
]

urls = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
pause_seconds = 0.1


def clean_columns(gdf):
    """Clean column names and ensure valid types for GeoJSON export."""
    if gdf.empty:
        return gdf

    # Clean names
    new_cols = []
    for c in gdf.columns:
        clean = re.sub(r"[^0-9a-zA-Z_]", "_", c)
        new_cols.append(clean[:60])
    gdf.columns = new_cols

    # Convert non-geometry columns to strings (GeoJSON accepts any type)
    for col in gdf.columns:
        if col != "geometry":
            try:
                gdf[col] = gdf[col].astype(str)
            except Exception:
                gdf[col] = gdf[col].apply(lambda x: str(x) if x is not None else "")
    return gdf


def query_overpass(bbox, queries, url, retries=3, pause_seconds=0.1):
    """Query Overpass API safely with retries.

    ``pause_seconds`` is taken as a parameter, not read from the module-level
    default, so a call from inside the thread pool in ``main`` never depends
    on a global being set before the pool starts.
    """
    q = """
    [out:json][timeout:180];
    ("""
    pairs = queries.items() if isinstance(queries, dict) else queries
    for key, value in pairs:
        if value:
            filter_expr = f'["{key}"="{value}"]'
        else:
            filter_expr = f'["{key}"]'
        q += f"""
        way{filter_expr}({bbox});
        relation{filter_expr}({bbox});
        """
    q += """
    );
    (._;>;);
    out body;
    """

    session = requests_session_with_retries(total_retries=0)
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params={"data": q}, timeout=300)

            if r.status_code == 200:
                try:
                    return r.json()
                except json.JSONDecodeError:
                    print("⚠️ JSON error → invalid response, skipping.")
                    return None
            elif r.status_code in [429, 504]:
                print(f"⚠️ Server limit ({r.status_code}) → waiting {pause_seconds * attempt}s...")
                time.sleep(pause_seconds * attempt)
            else:
                print(f"⚠️ Overpass error {r.status_code}")
                break
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            time.sleep(pause_seconds * attempt)
    return None


def elements_to_gdf(data):
    """Convert Overpass JSON to GeoDataFrames for lines and polygons only."""
    if not data or "elements" not in data:
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return empty.copy(), empty.copy()

    elements = data["elements"]
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in elements if el["type"] == "node"}
    ways = {el["id"]: el for el in elements if el["type"] == "way"}

    lns, polys = [], []

    for el in elements:
        tags = el.get("tags", {})
        if el["type"] == "way":
            coords = [nodes[n] for n in el.get("nodes", []) if n in nodes]
            if not coords:
                continue
            if coords[0] == coords[-1]:
                polys.append({**tags, "id": el["id"], "geometry": Polygon(coords)})
            else:
                lns.append({**tags, "id": el["id"], "geometry": LineString(coords)})
        elif el["type"] == "relation":
            outers = []
            for m in el.get("members", []):
                if m.get("type") == "way" and m.get("role") == "outer":
                    way_el = ways.get(m["ref"])
                    if way_el:
                        coords = [nodes[n] for n in way_el.get("nodes", []) if n in nodes]
                        if coords and coords[0] == coords[-1]:
                            outers.append(coords)
            if outers:
                try:
                    polys.append({**tags, "id": el["id"], "geometry": Polygon(outers[0])})
                except Exception:
                    pass

    gdf_lns = (
        gpd.GeoDataFrame(lns, geometry="geometry", crs="EPSG:4326")
        if lns
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    )
    gdf_poly = (
        gpd.GeoDataFrame(polys, geometry="geometry", crs="EPSG:4326")
        if polys
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    )
    return gdf_lns, gdf_poly


def timer(label):
    """Return a decorator that prints the runtime of the wrapped function."""

    def wrap(func):
        def inner(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            dt = time.time() - t0
            print(f"⏱️ {label}: {dt:.5f} sec")
            return result

        return inner

    return wrap


def find_bbox(geometry):
    """Return an Overpass bbox string for a geometry, or None when invalid."""
    if geometry is None or pd.isna(geometry):
        return None

    geometry = repair_geometry(geometry)
    if geometry is None:
        return None

    minx, miny, maxx, maxy = geometry.bounds
    return f"{miny},{minx},{maxy},{maxx}"


def create_tasks(gdf, batch_size, endpoint_urls=None):
    """Yield batched Overpass download tasks across the configured endpoints.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Grid features with ``bbox`` and ``idx`` columns.
    batch_size : int
        Number of requests to include in each yielded batch.

    Yields
    ------
    list[tuple[str, int, str]]
        Tuples of ``(bbox, idx, url)`` for one download batch.
    """
    if endpoint_urls is None:
        endpoint_urls = urls

    rows = list(gdf.itertuples())
    url_count = len(endpoint_urls)

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        yield [(r.bbox, r.idx, endpoint_urls[i % url_count]) for i, r in enumerate(batch)]


def row_operation(bbox, idx_val, url, output_folder, pause_seconds=0.1):
    """Fetch and persist OSM features for one grid-cell bbox.

    Parameters
    ----------
    bbox : str | None
        Overpass bbox string.
    idx_val : int
        Grid-cell identifier.
    url : str
        Overpass endpoint URL.
    output_folder : str
        Directory where per-tile GeoJSON outputs are written.
    pause_seconds : float
        Base backoff delay passed through to ``query_overpass``.
    """
    if bbox is None:
        return
    line_path = os.path.join(output_folder, f"idx_{idx_val}_lines.geojson")
    poly_path = os.path.join(output_folder, f"idx_{idx_val}_polygons.geojson")
    data = query_overpass(bbox, queries, url, pause_seconds=pause_seconds)
    all_lines, all_polys = elements_to_gdf(data)

    if not all_lines.empty:
        all_lines = clean_columns(all_lines)

        with atomic_output(line_path) as tmp_path:
            all_lines.to_file(tmp_path, driver="GeoJSON")
        print(f"✅ Saved {line_path}")

    if not all_polys.empty:
        all_polys = clean_columns(all_polys)
        poly_path = os.path.join(output_folder, f"idx_{idx_val}_polygons.geojson")
        with atomic_output(poly_path) as tmp_path:
            all_polys.to_file(tmp_path, driver="GeoJSON")
        print(f"✅ Saved {poly_path}")


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run NEW_02_EXTRACTOSMDATAFULL_GEOJSON.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Load annotation grids, query Overpass, and persist per-tile OSM outputs.

    Returns
    -------
    None
        The function writes per-grid polygon and line GeoJSON files to disk.
    """
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="NEW_02_EXTRACTOSMDATAFULL_GEOJSON", **overrides)
    global pause_seconds
    global urls
    overwrite = cfg["overwrite_existing"]
    retrials = int(cfg["annotations"]["retries"])
    urls = list(cfg["annotations"]["overpass_urls"])
    pause_seconds = float(cfg["annotations"]["overpass_pause_seconds"])

    points_path = cfg["paths"]["corrected_all_filepath"]

    grid_filedir = cfg["paths"]["annotations_grid_dir"]
    grid_filepath = os.path.join(grid_filedir, f"grids_{os.path.basename(points_path)}")
    output_folder = cfg["paths"]["annotations_by_osm_dir"]

    os.makedirs(output_folder, exist_ok=True)
    max_workers = int(cfg["annotations"]["max_workers"])
    batch_size = max_workers * len(urls)

    poly = gpd.read_file(grid_filepath).to_crs(4326)
    poly["bbox"] = poly["geometry"].map(find_bbox)

    for _ in range(retrials):
        if not overwrite:
            output_list = set(os.listdir(output_folder))
            mask = []
            for idx in poly["idx"].values:
                line_file = f"idx_{idx}_lines.geojson"
                poly_file = f"idx_{idx}_polygons.geojson"
                if line_file in output_list or poly_file in output_list:
                    mask.append(False)
                else:
                    mask.append(True)
            poly = poly[mask]
            if poly.empty:
                print("🎉 All idx features already processed, no retrials needed!")
                return

        print(len(poly))
        print(poly.head())

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for tasks in create_tasks(poly, batch_size, urls):
                futures = [
                    executor.submit(row_operation, bbox, idx_val, url, output_folder, pause_seconds)
                    for bbox, idx_val, url in tasks
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"⚠️ Error processing task: {e}")
            print("\n🎯 All requested idx features processed successfully!")


if __name__ == "__main__":
    configure_logging()
    main()
