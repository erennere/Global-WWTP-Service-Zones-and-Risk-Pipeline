"""Reference data and national plant registers: download, describe, and list the plants our list lacks.

Needs internet for ``fetch`` (run it where the cluster has it; the Snakemake rule is a local rule).
Settings: config.yaml, section ``national_data``.

Commands (``python -m wwtp.pop_validation_scripts.national_data <command>``)::

    fetch      every file of ``sources`` into <validation_dir>/<source>/ (files already there are kept);
               manifest.json records URL, bytes, sha256 or the error per file. A failed download of a
               ``required`` source fails the run; the others are only recorded.
    registers  peek.md: structure of every downloaded file of the sources with ``peek`` (zip members,
               layers, columns, first rows), so that parsers can be written without the data at hand.
               register_plants.gpkg / .csv: every plant of the ``registers`` as a point with its original
               attributes (JSON), its distance to the nearest plant of ours and that plant's coordinates.
               new_plants.gpkg / new_plants.csv: the register plants farther than ``match_distance_m``
               from every plant of ours - one file for all countries.
               uncovered_plants.gpkg / .csv: the countries our list does not cover (``uncovered``):
               their register plants and HydroWASTE records in one file, without size; not merged
               into our plant list.
               registers_summary.csv, and results.zip with the small files to send back.

File kinds: ``http`` (one GET), ``ogcapi`` (OGC API Features: pages of ``page_size`` followed through the
``next`` links, saved as one GeoJSON), ``arcgis`` (ArcGIS REST layer: ``query`` in pages, saved as one
GeoJSON).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import zipfile
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy.spatial import cKDTree

from ..geo_utils import read_located_plants
from ..starter import load_config
from ..utils import atomic_output, configure_logging, get_iso_codes, requests_session_with_retries

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_008.8
SPATIAL = (".shp", ".geojson", ".json", ".gpkg", ".gml", ".kml")
TABLES = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
ZIP_LISTING = 60  # members shown per archive in peek.md


# --------------------------------------------------------------------------- #
# fetch                                                                       #
# --------------------------------------------------------------------------- #
def source_path(cfg, source, name):
    return os.path.join(cfg["paths"]["validation_dir"], source, name)


def _get_json(session, url, params, timeout):
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_http(session, url, target, timeout):
    """One GET, streamed to ``target``. Returns the final URL and the content type."""
    with session.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with open(target, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        return response.url, response.headers.get("content-type", "")


def fetch_ogcapi(session, url, target, timeout, page_size):
    """Every feature of an OGC API Features collection (``.../items``), as one GeoJSON."""
    features, page, params = [], url, {"f": "json", "limit": page_size}
    while page:
        data = _get_json(session, page, params, timeout)
        features += data["features"]
        page = next((link["href"] for link in data["links"] if link["rel"] == "next"), None)
        params = None  # the next link carries its own query
    with open(target, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh)
    return url, "application/geo+json"


def fetch_arcgis(session, url, target, timeout, page_size):
    """Every feature of an ArcGIS REST layer (``.../MapServer/<n>``), as one GeoJSON in EPSG:4326."""
    features, offset = [], 0
    while True:
        params = {
            "where": "1=1",
            "outFields": "*",
            "outSR": 4326,
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        batch = _get_json(session, f"{url}/query", params, timeout)["features"]
        features += batch
        offset += len(batch)
        if len(batch) < page_size:
            break
    with open(target, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh)
    return url, "application/geo+json"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_all(cfg):
    """Download every source; returns the manifest. Raises when a required file is missing afterwards."""
    root = cfg["paths"]["validation_dir"]
    os.makedirs(root, exist_ok=True)
    manifest_path = os.path.join(root, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    session = requests_session_with_retries()
    session.headers.update(cfg["request_headers"])  # some portals refuse the default python-requests agent
    timeout, page_size = cfg["request_timeout_seconds"], int(cfg["page_size"])
    missing = []
    for source, entry in cfg["sources"].items():
        os.makedirs(os.path.join(root, source), exist_ok=True)
        for name, spec in entry["files"].items():
            target, key = source_path(cfg, source, name), f"{source}/{name}"
            if os.path.exists(target) and os.path.getsize(target) > 0:
                if key not in manifest or "error" in manifest[key]:
                    manifest[key] = {"url": spec["url"], "bytes": os.path.getsize(target), "note": "already there"}
                continue
            record = {"url": spec["url"], "kind": spec["kind"], "tried_utc": datetime.now(timezone.utc).isoformat()}
            try:
                with atomic_output(target) as tmp:
                    if spec["kind"] == "http":
                        final_url, content_type = fetch_http(session, spec["url"], tmp, timeout)
                    elif spec["kind"] == "ogcapi":
                        final_url, content_type = fetch_ogcapi(session, spec["url"], tmp, timeout, page_size)
                    else:
                        final_url, content_type = fetch_arcgis(session, spec["url"], tmp, timeout, page_size)
                    record.update(final_url=final_url, content_type=content_type, sha256=_sha256(tmp))
                record["bytes"] = os.path.getsize(target)
                logger.info("%s: %d bytes", key, record["bytes"])
            except Exception as err:  # every source is tried; the failures are listed at the end
                record["error"] = f"{type(err).__name__}: {err}"[:500]
                logger.warning("%s failed: %s", key, record["error"])
                if entry["required"]:
                    missing.append(key)
            manifest[key] = record
            with atomic_output(manifest_path) as tmp, open(tmp, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=1, sort_keys=True)
    with atomic_output(manifest_path) as tmp, open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    if missing:
        raise RuntimeError(f"required reference files could not be downloaded: {missing} (see {manifest_path})")
    return manifest


# --------------------------------------------------------------------------- #
# peek                                                                        #
# --------------------------------------------------------------------------- #
def _peek_table(source, name, rows):
    """First rows of a table (a path or the bytes of a zip member), without assuming where its header is."""
    if name.lower().endswith((".xlsx", ".xls")):
        book = pd.ExcelFile(source)
        out = [f"sheets: {book.sheet_names}"]
        for sheet in book.sheet_names[:12]:
            head = book.parse(sheet, nrows=rows + 3, header=None)
            out += [f"sheet '{sheet}':", head.astype(str).to_csv(index=False, header=False, sep="|").strip()]
        return "\n".join(out)
    if isinstance(source, io.BytesIO):
        raw = source.read(200_000)
    else:
        with open(source, "rb") as fh:
            raw = fh.read(200_000)
    return "\n".join(raw.decode("utf-8", errors="replace").splitlines()[: rows + 3])


def _peek_spatial(path, rows):
    out = []
    for layer, geometry in pyogrio.list_layers(path):
        info = pyogrio.read_info(path, layer=layer)
        out.append(
            f"layer '{layer}': {info['features']} features, {geometry}, crs {info['crs']}, "
            f"fields {dict(zip(info['fields'].tolist(), info['dtypes'].tolist()))}"
        )
        head = pyogrio.read_dataframe(path, layer=layer, max_features=rows, read_geometry=False)
        out.append(head.astype(str).to_csv(index=False, sep="|").strip())
    return "\n".join(out)


def peek_file(path, rows):
    """A text description of one downloaded file (never raises: the error becomes the description)."""
    lines = [f"### {os.path.basename(path)} ({os.path.getsize(path):,} bytes)"]
    try:
        lower = path.lower()
        if zipfile.is_zipfile(path) and not lower.endswith((".xlsx", ".gpkg")):
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                lines.append(f"zip with {len(members)} members:")
                lines += [f"- {m.filename} ({m.file_size:,})" for m in members[:ZIP_LISTING]]
                for m in members:
                    inner = m.filename.lower()
                    if inner.endswith((".shp", ".geojson", ".gpkg", ".gml")):
                        lines += [f"#### {m.filename}", _peek_spatial(f"/vsizip/{path}/{m.filename}", rows)]
                    elif inner.endswith(TABLES):
                        with archive.open(m) as fh:
                            lines += [f"#### {m.filename}", _peek_table(io.BytesIO(fh.read(20_000_000)), inner, rows)]
        elif lower.endswith(SPATIAL):
            lines.append(_peek_spatial(path, rows))
        elif lower.endswith(TABLES):
            lines.append(_peek_table(path, lower, rows))
        else:
            lines.append("(not described: unknown type)")
    except Exception as err:  # a description tool: report and go on
        lines.append(f"could not be described: {type(err).__name__}: {err}"[:500])
    return "\n".join(lines)


def peek_all(cfg):
    rows = int(cfg["peek_rows"])
    out = ["# Downloaded files", ""]
    for source, entry in cfg["sources"].items():
        if not entry["peek"]:
            continue
        out.append(f"## {source}")
        for name in entry["files"]:
            path = source_path(cfg, source, name)
            out.append(peek_file(path, rows) if os.path.exists(path) else f"### {name}: not downloaded")
            out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# registers                                                                   #
# --------------------------------------------------------------------------- #
def sphere_xyz(lon, lat):
    """Points on a sphere of the Earth's radius; chord lengths equal ground distances at these scales."""
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return EARTH_RADIUS_M * np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _coordinate_column(columns, tokens):
    """The first column whose lower-case name contains one of ``tokens`` (in the order of the tokens)."""
    for token in tokens:
        for column in columns:
            if token in str(column).lower():
                return column
    raise KeyError(f"no column with one of {tokens} among {list(columns)}")


def _points_from_table(frame, names):
    lon = pd.to_numeric(frame[_coordinate_column(frame.columns, names["lon"])], errors="coerce")
    lat = pd.to_numeric(frame[_coordinate_column(frame.columns, names["lat"])], errors="coerce")
    ok = lon.between(-180, 180) & lat.between(-90, 90)
    frame = frame[ok].reset_index(drop=True)
    return gpd.GeoDataFrame(frame, geometry=gpd.points_from_xy(lon[ok], lat[ok]), crs=4326).assign(layer="")


def _spatial_layers(path):
    """Every vector layer of a file or of the spatial members of a zip, as (label, readable path, layer)."""
    lower = path.lower()
    if zipfile.is_zipfile(path) and not lower.endswith(".gpkg"):
        with zipfile.ZipFile(path) as archive:
            members = sorted(m for m in archive.namelist() if m.lower().endswith((".shp", ".geojson", ".gpkg", ".gml")))
        files = [(m, f"/vsizip/{path}/{m}") for m in members]
    else:
        files = [("", path)]
    return [
        (f"{label}:{layer}".strip(":"), file, layer) for label, file in files for layer, _ in pyogrio.list_layers(file)
    ]


def read_register(cfg, register):
    """A register's plants as points in EPSG:4326 with ``layer`` and the original attributes as JSON."""
    entry = cfg["registers"][register]
    path = source_path(cfg, entry["source"], entry["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"not downloaded: {path}")
    lower = path.lower()
    if lower.endswith(TABLES):
        if lower.endswith((".xlsx", ".xls")):
            table = pd.read_excel(path)
        else:
            table = pd.read_csv(path, low_memory=False, encoding=entry["encoding"])
        parts = [_points_from_table(table, cfg["coordinate_names"])]
    else:
        parts = []
        for label, file, layer in _spatial_layers(path):
            part = gpd.read_file(file, layer=layer)
            part = part[part.geometry.notna() & ~part.geometry.is_empty]
            if part.crs is None:
                logger.warning("%s %s has no CRS: taken as longitude / latitude", register, label)
                part = part.set_crs(4326)
            parts.append(part.to_crs(4326).assign(layer=label))
    rows = []
    for part in parts:
        points = part.geometry.representative_point()
        attrs = pd.DataFrame(part.drop(columns=["geometry", "layer"])).astype(str)
        rows.append(
            gpd.GeoDataFrame(
                {
                    "register": register,
                    "iso2": entry["iso2"],
                    "year": entry["year"],
                    "layer": part["layer"].to_numpy(),
                    "lon": points.x.to_numpy(),
                    "lat": points.y.to_numpy(),
                    "attrs": [json.dumps(r, ensure_ascii=False) for r in attrs.to_dict("records")],
                },
                geometry=points.to_numpy(),
                crs=4326,
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_uncovered(cfg, registers):
    """Plants of the countries our list does not cover: register plants and HydroWASTE records, one file.

    ``uncovered.countries`` (ISO 3166 alpha-2): every register plant of these countries, and every
    HydroWASTE record (``paths.hydrowaste``, columns ``uncovered.hydrowaste_columns``) of them. A
    HydroWASTE record within ``match_distance_m`` of a register plant of the same country is the same
    plant (one row, both sources). No size: the original attributes are kept as JSON, unused.
    """
    spec, limit = cfg["uncovered"], float(cfg["match_distance_m"])
    _, alpha_2_to_3, _, _ = get_iso_codes()
    col = spec["hydrowaste_columns"]
    hw = pd.read_csv(cfg["paths"]["hydrowaste"], low_memory=False)
    rows = []
    for iso2 in spec["countries"]:
        reg = registers[registers["iso2"] == iso2].reset_index(drop=True)
        h = hw[hw[col["country"]] == alpha_2_to_3[iso2]].reset_index(drop=True)
        h = h[
            pd.to_numeric(h[col["lon"]], errors="coerce").notna()
            & pd.to_numeric(h[col["lat"]], errors="coerce").notna()
        ]
        h = h.reset_index(drop=True)
        hw_of = np.full(len(reg), -1)
        taken = np.zeros(len(h), dtype=bool)
        if len(reg) and len(h):
            d, i = cKDTree(sphere_xyz(h[col["lon"]].astype(float), h[col["lat"]].astype(float))).query(
                sphere_xyz(reg["lon"], reg["lat"])
            )
            for r in np.argsort(d):  # nearest pairs first, one to one
                if d[r] <= limit and not taken[i[r]]:
                    hw_of[r], taken[i[r]] = i[r], True
        hw_attrs = [json.dumps(rec, ensure_ascii=False, default=str) for rec in h.astype(str).to_dict("records")]
        for r, plant in reg.iterrows():
            j = hw_of[r]
            rows.append(
                {
                    "iso2": iso2,
                    "sources": "register+hydrowaste" if j >= 0 else "register",
                    "register": plant["register"],
                    "lon": plant["lon"],
                    "lat": plant["lat"],
                    "register_attrs": plant["attrs"],
                    "hydrowaste_id": h[col["id"]].iloc[j] if j >= 0 else None,
                    "hydrowaste_name": h[col["name"]].iloc[j] if j >= 0 else None,
                    "hydrowaste_attrs": hw_attrs[j] if j >= 0 else None,
                }
            )
        for j in np.flatnonzero(~taken):
            rows.append(
                {
                    "iso2": iso2,
                    "sources": "hydrowaste",
                    "register": None,
                    "lon": float(h[col["lon"]].iloc[j]),
                    "lat": float(h[col["lat"]].iloc[j]),
                    "register_attrs": None,
                    "hydrowaste_id": h[col["id"]].iloc[j],
                    "hydrowaste_name": h[col["name"]].iloc[j],
                    "hydrowaste_attrs": hw_attrs[j],
                }
            )
        logger.info(
            "uncovered %s: %d register plants, %d HydroWASTE records, %d matched",
            iso2,
            len(reg),
            len(h),
            int(taken.sum()),
        )
    columns = ["iso2", "sources", "register", "lon", "lat", "register_attrs", "hydrowaste_id", "hydrowaste_name"]
    out = pd.DataFrame(rows, columns=columns + ["hydrowaste_attrs"])
    return gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["lon"], out["lat"]), crs=4326)


def build_registers(cfg):
    """peek.md, register_plants, new_plants, uncovered_plants (.gpkg / .csv), registers_summary.csv, results.zip."""
    out = cfg["paths"]["output_dir"]
    os.makedirs(out, exist_ok=True)
    ours = read_located_plants(cfg["paths"]["annotated_all_filepath"]).to_crs(4326)
    points = ours.geometry.representative_point()
    tree = cKDTree(sphere_xyz(points.x, points.y))
    limit = float(cfg["match_distance_m"])
    plants, summary = [], []
    for register in cfg["registers"]:
        try:
            part = read_register(cfg, register)
        except Exception as err:  # one unreadable register must not hide the others
            summary.append({"register": register, "status": f"{type(err).__name__}: {err}"[:300]})
            logger.warning("%s: %s", register, summary[-1]["status"])
            continue
        distance, nearest = tree.query(sphere_xyz(part["lon"], part["lat"]))
        part["distance_to_ours_m"] = distance
        part["new"] = distance > limit
        # the nearest of our plants, by its coordinates (the key to its footprint and sizes)
        part["ours_lon"], part["ours_lat"] = points.x.to_numpy()[nearest], points.y.to_numpy()[nearest]
        plants.append(part)
        summary.append(
            {
                "register": register,
                "status": "ok",
                "plants": len(part),
                "new": int(part["new"].sum()),
                "median_distance_to_ours_m": float(np.median(distance)),
                "layers": ";".join(sorted(set(part["layer"]))),
            }
        )
    summary = pd.DataFrame(summary)
    columns = ["register", "iso2", "year", "layer", "lon", "lat", "attrs", "distance_to_ours_m", "new"]
    columns += ["ours_lon", "ours_lat", "geometry"]
    table = (
        pd.concat(plants, ignore_index=True)
        if plants
        else gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=4326)
    )
    new = table[table["new"].astype(bool)]
    uncovered = build_uncovered(cfg, table)
    files = {}
    for name, frame in (
        ("register_plants.gpkg", table),
        ("new_plants.gpkg", new),
        ("uncovered_plants.gpkg", uncovered),
    ):
        with atomic_output(os.path.join(out, name)) as tmp:
            frame.to_file(tmp, driver="GPKG")
    for name, text in (
        ("new_plants.csv", pd.DataFrame(new.drop(columns="geometry")).to_csv(index=False)),
        ("register_plants.csv", pd.DataFrame(table.drop(columns="geometry")).to_csv(index=False)),
        ("uncovered_plants.csv", pd.DataFrame(uncovered.drop(columns="geometry")).to_csv(index=False)),
        ("registers_summary.csv", summary.to_csv(index=False)),
        ("peek.md", peek_all(cfg)),
    ):
        files[name] = text
        with atomic_output(os.path.join(out, name)) as tmp, open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
    manifest = os.path.join(cfg["paths"]["validation_dir"], "manifest.json")
    with atomic_output(os.path.join(out, "results.zip")) as tmp, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
        if os.path.exists(manifest):
            z.write(manifest, "manifest.json")
    logger.info("registers: %d plants, %d new (farther than %.0f m from ours)", len(table), len(new), limit)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["fetch", "registers"])
    args = parser.parse_args(argv)
    cfg = load_config(script_name="national_data")
    if args.command == "fetch":
        fetch_all(cfg)
    else:
        build_registers(cfg)


if __name__ == "__main__":
    configure_logging()
    main()
