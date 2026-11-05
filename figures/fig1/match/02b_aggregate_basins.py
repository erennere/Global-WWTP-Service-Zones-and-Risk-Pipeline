"""Aggregate reconciled plants to HydroBASINS level 3 and level 4.

Basin of a plant:
  * HeiGIT plants that carry a level-9 HYBAS_ID: PFAF_ID of that level-9 basin, truncated to 3 / 4 digits
    (Pfafstetter codes are nested: the first k digits of a level-9 code are its level-k ancestor).
  * all other plants (reference-only 'missing' plants, HeiGIT plants without HYBAS_ID): point-in-polygon
    on the level-3 / level-4 polygons, nearest polygon within 25 km for coastal outfalls.
  The Pfafstetter route is cross-checked against point-in-polygon for every HeiGIT plant.
Plants located in countries without HeiGIT coverage are kept out of the counts; a basin whose reference
plants lie mostly in such countries is flagged 'uncovered'.

Usage: python 02b_aggregate_basins.py <hydrobasins_dir> <agg_dir> [radius_tag]
  <hydrobasins_dir> holds hybas_lev03_v1c.gpkg, hybas_lev04_v1c.gpkg and hybas_lev09_v1c_attributes.csv
"""
import sys, json
import numpy as np, pandas as pd, geopandas as gpd
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig1_config import MATCH_RADIUS_M, MIN_PLANTS, PATHS, TAG  # noqa: E402  (config.yaml, section figure1)

HB, AGG = sys.argv[1:3]
TAG = sys.argv[3] if len(sys.argv) > 3 else TAG
SIMPLIFY_M = 3000   # plotting only; 1 px at 600 dpi on an 88 mm global map is ~16 km

pts = pd.read_csv(f"{AGG}/plants_with_units_{TAG}.csv", keep_default_na=False, na_values=[""])
ref = pd.read_csv(f"{AGG}/reference_with_units_{TAG}.csv", keep_default_na=False, na_values=[""])
cty = gpd.read_file(f"{AGG}/units_country_{TAG}.gpkg", ignore_geometry=True)
unc_cty = set(cty.loc[cty.status == "uncovered", "iso2"])
pts["in_unc"] = pts.unit.isin(unc_cty); ref["in_unc"] = ref.unit.isin(unc_cty)

att9 = pd.read_csv(f"{HB}/hybas_lev09_v1c_attributes.csv")
pfaf9 = dict(zip(att9.HYBAS_ID.astype("int64"), att9.PFAF_ID.astype("int64")))


def spatial(df, polys, key):
    g = gpd.GeoDataFrame(df[["lon", "lat"]].copy(), geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
    j = gpd.sjoin(g, polys[[key, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    miss = j[key].isna()
    if miss.any():
        nn = gpd.sjoin_nearest(g[miss].to_crs(8857), polys[[key, "geometry"]].to_crs(8857), how="left",
                               max_distance=25000)
        nn = nn[~nn.index.duplicated()]
        j.loc[miss, key] = nn[key].values
    return j[key].values


def aggregate(p, r, units, key):
    a = pd.DataFrame({
        "n_heigit": p[p.cls != "missing"].groupby("unit").size(),
        "n_new": p[p.cls == "new"].groupby("unit").size(),
        "n_corr": p[p.cls == "corrected"].groupby("unit").size(),
        "n_match": p[p.cls == "matched"].groupby("unit").size(),
        "med_shift": p[p.cls == "corrected"].groupby("unit").shift_m.median(),
        "q25_shift": p[p.cls == "corrected"].groupby("unit").shift_m.quantile(.25),
        "q75_shift": p[p.cls == "corrected"].groupby("unit").shift_m.quantile(.75),
        "med_shift_all": p[p.cls.isin(["corrected", "matched"])].groupby("unit").shift_m.median(),
    })
    rr = pd.DataFrame({"n_ref": r.groupby("unit").size(), "n_missing": r[r.cls == "missing"].groupby("unit").size()})
    a = a.join(rr, how="outer")
    cnt = [c for c in a.columns if c.startswith("n_")]
    a[cnt] = a[cnt].fillna(0).astype(int)
    a["p_new"] = a.n_new / a.n_heigit.replace(0, np.nan)
    a["p_missing"] = a.n_missing / a.n_ref.replace(0, np.nan)
    a["status"] = np.select([a.n_ref == 0, a.n_heigit == 0], ["no_ref", "uncovered_tmp"], "ok")
    out = units.merge(a, left_on=key, right_index=True, how="left")
    out["status"] = out["status"].fillna("no_ref")
    out[cnt] = out[cnt].fillna(0).astype(int)
    return out


report = {}
for lev in (3, 4):
    polys = gpd.read_file(f"{HB}/hybas_lev{lev:02d}_v1c.gpkg")[["HYBAS_ID", "PFAF_ID", "SUB_AREA", "geometry"]]
    polys["PFAF_ID"] = polys.PFAF_ID.astype("int64")
    # the HydroBASINS lake version stores lakes as separate polygons sharing their basin's PFAF_ID
    polys["geometry"] = polys.geometry.make_valid()
    polys = polys.dissolve(by="PFAF_ID", aggfunc={"HYBAS_ID": "first", "SUB_AREA": "sum"}, as_index=False)
    key = "PFAF_ID"
    # --- HeiGIT plants: Pfafstetter route, cross-checked
    h = pts[pts.cls != "missing"].copy()
    hb = pd.to_numeric(h.HYBAS_ID, errors="coerce")
    pf = hb.map(lambda v: pfaf9.get(int(v)) if pd.notna(v) else None)
    h["pfaf_route"] = pf.map(lambda v: int(str(int(v))[:lev]) if pd.notna(v) else np.nan)
    h["spatial_route"] = spatial(h, polys, key)
    both = h.pfaf_route.notna() & h.spatial_route.notna()
    agree = float((h.loc[both, "pfaf_route"] == h.loc[both, "spatial_route"]).mean())
    h["unit"] = h.pfaf_route.where(h.pfaf_route.notna(), h.spatial_route)
    # --- reference-only plants and the reference table: spatial
    m = pts[pts.cls == "missing"].copy(); m["unit"] = spatial(m, polys, key)
    r = ref.copy(); r["unit"] = spatial(r, polys, key)
    p = pd.concat([h, m], ignore_index=True)
    p_cov, r_cov = p[~p.in_unc], r[~r.in_unc]
    units = aggregate(p_cov, r_cov, polys, key)
    # basins whose reference plants are mostly in uncovered countries
    share_unc = r.groupby("unit").in_unc.mean()
    units["share_ref_uncovered"] = units[key].map(share_unc).fillna(0)
    units.loc[(units.share_ref_uncovered >= 0.5) & (units.n_heigit == 0), "status"] = "uncovered"
    units.loc[units.status == "uncovered_tmp", "status"] = "ok"   # covered basin, all reference plants missing
    units["geometry"] = units.to_crs(8857).geometry.simplify(SIMPLIFY_M).make_valid().to_crs(4326).make_valid().values
    units.to_file(f"{AGG}/units_hybas{lev}_{TAG}.gpkg", driver="GPKG")
    ok = units[units.status == "ok"]
    report[f"lev{lev}"] = dict(
        basins=len(units), ok=len(ok), uncovered=int((units.status == "uncovered").sum()),
        no_ref=int((units.status == "no_ref").sum()),
        heigit_with_hybas=int(h.pfaf_route.notna().sum()), pfaf_vs_spatial_agreement=round(agree, 4),
        unassigned_heigit=int(h.unit.isna().sum()), unassigned_missing=int(m.unit.isna().sum()),
        n_ref_lt5=int((ok.n_ref < 5).sum()), n_corr_lt5=int((ok.n_corr < 5).sum()),
        totals=dict(ref=int(units.n_ref.sum()), heigit=int(units.n_heigit.sum()), new=int(units.n_new.sum()),
                    missing=int(units.n_missing.sum())),
    )
    print(json.dumps(report[f"lev{lev}"], indent=1))
json.dump(report, open(f"{AGG}/basin_aggregation_report_{TAG}.json", "w"), indent=1)
