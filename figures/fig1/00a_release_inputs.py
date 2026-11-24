#!/usr/bin/env python3
"""Figure 1 inputs from a release file (HeiGIT_WWTP_<tag>.gpkg, wwtp/release_export.py) instead of the first release.

Three steps, each a subcommand; run before 00_fix_utm.py and `snakemake fig1`:

  plants <release.gpkg> <match_data_dir>
      From the points layer (WWTP_LOCATIONS; the service-area polygons are not read):
        heigit_points.csv  fid, lon, lat, WASTE_ID, HW_WASTE_ID, uwwCode, ISO_2, CNTRY_ISO, HYBAS_ID, SOURCE_DS,
                           num_detections, category_name, POP_SERVED  (01_reconcile, 00_fix_utm)
        heigit_epsg.csv    fid, epsg  (00_fix_utm's report)
        heigit_attrs.csv   fid, ISO_2, round_area, rect_area, category_number, category_name, SOURCE_DS
                           (the tanks panel: figure1.paths.heigit_attrs)
        heigit_pop.csv     fid, zonal_sum_2024, zonal_sum_2020  (05_service, 05b_service_country)
      fid is the release file's fid. A heigit_points_original.csv left by an earlier 00_fix_utm run is removed,
      so 00_fix_utm starts again from these plants. After 00_fix_utm, copy heigit_points.csv and heigit_pop.csv
      to <cartogram_dir>/data.

  non_served <non_served_areas_above_threshold.gpkg> <out.csv>
      The non-served urban areas (find_unserved_pop) as one point per area (representative point, WGS84):
      fid, HYBAS_ID, data_idx, tile, country, pop_sum, lon, lat (the columns the layer has). Exact duplicate
      rows (same attributes and geometry) are dropped and counted. Snakemake: fig1_non_served_release, for the
      variant release_export exports. Copy the CSV to <cartogram_dir>/data/non_served_points.csv.

  basins <hydrobase_lvl<N>_combined.gpkg> <hydrobasins_dir>
      hybas_lev<NN>_pipeline_attributes.csv (HYBAS_ID, PFAF_ID, SUB_AREA) of the pipeline's basins, so that the
      HYBAS_IDs of the release (level of the variant, standard HydroBASINS) are found by fig1_config.pfaf_lookup.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyogrio

POINT_COLUMNS = ["WASTE_ID", "HW_WASTE_ID", "uwwCode", "ISO_2", "CNTRY_ISO", "HYBAS_ID", "SOURCE_DS",
                 "num_detections", "category_name", "POP_SERVED"]
ATTR_COLUMNS = ["ISO_2", "round_area", "rect_area", "category_number", "category_name", "SOURCE_DS"]
POP_YEARS = (2024, 2020)
NON_SERVED_COLUMNS = ["HYBAS_ID", "NEXT_DOWN", "MAIN_BAS", "data_idx", "tile", "country", "pop_sum"]


def plants(release, out_dir, layer="WWTP_LOCATIONS"):
    pop_cols = [f"{y}_zonal_sum" for y in POP_YEARS]
    want = list(dict.fromkeys(POINT_COLUMNS + ATTR_COLUMNS + ["epsg"] + pop_cols))
    have = set(pyogrio.read_info(release, layer=layer)["fields"])
    missing = [c for c in want if c not in have]
    assert not missing, f"{release} ({layer}) lacks {missing}"
    g = pyogrio.read_dataframe(release, layer=layer, columns=want, fid_as_index=True)
    assert g.crs is not None and g.crs.to_epsg() == 4326, g.crs
    assert g.geometry.notna().all() and (g.geometry.geom_type == "Point").all()
    g.index.name = "fid"
    g = g.reset_index()
    hw = pd.to_numeric(g.HW_WASTE_ID, errors="coerce")  # text in the release ('54120.0')
    assert (hw.dropna() % 1 == 0).all(), "HW_WASTE_ID that is not a whole number"
    pts = pd.DataFrame({"fid": g.fid, "lon": g.geometry.x.round(6), "lat": g.geometry.y.round(6)})
    for c in POINT_COLUMNS:
        pts[c] = g[c]
    pts["HW_WASTE_ID"] = hw.astype("Int64")
    pts["HYBAS_ID"] = pd.to_numeric(g.HYBAS_ID, errors="coerce").astype("Int64")
    os.makedirs(out_dir, exist_ok=True)
    stale = os.path.join(out_dir, "heigit_points_original.csv")
    if os.path.exists(stale):
        os.remove(stale)  # 00_fix_utm keeps the unrepaired plants there; they belong to the old file
    pts.to_csv(os.path.join(out_dir, "heigit_points.csv"), index=False)
    g[["fid", "epsg"]].to_csv(os.path.join(out_dir, "heigit_epsg.csv"), index=False)
    g[["fid", *ATTR_COLUMNS]].to_csv(os.path.join(out_dir, "heigit_attrs.csv"), index=False)
    pop = pd.DataFrame({"fid": g.fid, **{f"zonal_sum_{y}": g[f"{y}_zonal_sum"] for y in POP_YEARS}})
    pop.to_csv(os.path.join(out_dir, "heigit_pop.csv"), index=False)
    report = {
        "release": os.path.abspath(release),
        "plants": len(pts),
        "countries": int(pts.ISO_2.nunique()),
        "with_HW_WASTE_ID": int(pts.HW_WASTE_ID.notna().sum()),
        "with_uwwCode": int(pts.uwwCode.notna().sum()),
        "without_HYBAS_ID": int(pts.HYBAS_ID.isna().sum()),
        "HYBAS_ID_levels": sorted({int(str(int(v))[1:3]) for v in pts.HYBAS_ID.dropna()}),
        "by_SOURCE_DS": pts.SOURCE_DS.value_counts(dropna=False).to_dict(),
        "served_2024": float(pop.zonal_sum_2024.sum()),
        "plants_without_served_2024": int(pop.zonal_sum_2024.isna().sum()),
    }
    with open(os.path.join(out_dir, "release_inputs_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))


def non_served(layer_path, out_csv):
    g = pyogrio.read_dataframe(layer_path)
    assert g.crs is not None, f"{layer_path} has no CRS"
    g = g.to_crs(4326)
    cols = [c for c in NON_SERVED_COLUMNS if c in g.columns]
    assert "pop_sum" in cols, f"{layer_path} has no pop_sum"
    key = g[cols].astype(str).agg("|".join, axis=1) + "|" + g.geometry.to_wkb().map(bytes.hex)
    dup = key.duplicated()
    n_rows, pop_rows = len(g), float(g.pop_sum.sum())
    g = g[~dup].reset_index(drop=True)
    pt = g.geometry.representative_point()
    out = pd.DataFrame({"fid": np.arange(1, len(g) + 1)})
    for c in cols:
        out[c] = g[c].values
    out["lon"], out["lat"] = pt.x.round(6).values, pt.y.round(6).values
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    out.to_csv(out_csv, index=False)
    report = {
        "layer": os.path.abspath(layer_path),
        "rows": n_rows,
        "exact_duplicates_dropped": int(dup.sum()),
        "areas": len(out),
        "pop_sum_rows": pop_rows,
        "pop_sum_areas": float(out.pop_sum.sum()),
        "columns": list(out.columns),
    }
    with open(os.path.splitext(out_csv)[0] + "_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


def basins(watershed, hb_dir):
    t = pyogrio.read_dataframe(watershed, columns=["HYBAS_ID", "PFAF_ID", "SUB_AREA"], read_geometry=False)
    t = t.astype({"HYBAS_ID": "int64", "PFAF_ID": "int64"})
    levels = {int(str(v)[1:3]) for v in t.HYBAS_ID}
    assert len(levels) == 1, f"{watershed} mixes HydroBASINS levels {sorted(levels)}"
    out = os.path.join(hb_dir, f"hybas_lev{levels.pop():02d}_pipeline_attributes.csv")
    os.makedirs(hb_dir, exist_ok=True)
    t.drop_duplicates("HYBAS_ID").to_csv(out, index=False)
    print(f"wrote {out}: {len(t):,} basins")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plants")
    p.add_argument("release")
    p.add_argument("out_dir")
    p.add_argument("--layer", default="WWTP_LOCATIONS")
    n = sub.add_parser("non_served")
    n.add_argument("layer_path")
    n.add_argument("out_csv")
    b = sub.add_parser("basins")
    b.add_argument("watershed")
    b.add_argument("hb_dir")
    a = ap.parse_args(argv)
    if a.cmd == "plants":
        plants(a.release, a.out_dir, a.layer)
    elif a.cmd == "non_served":
        non_served(a.layer_path, a.out_csv)
    else:
        basins(a.watershed, a.hb_dir)


if __name__ == "__main__":
    sys.exit(main())
