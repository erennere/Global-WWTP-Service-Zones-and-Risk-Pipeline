"""Served share of the urban population per HydroBASINS level 3 / level 4 basin (Figure 1b).

  P_served = sum of HeiGIT service-area population (2024_zonal_sum) of the plants in the basin
  P_non    = sum of pop_sum of non-served urban areas (non_served_nxt_river_v2_lvl8_bfk0_8_log_mult,
             exact duplicate rows removed upstream) in the basin
  S        = P_served / (P_served + P_non)

Basin of a non-served area: level-8 HYBAS_ID -> PFAF_ID (HydroBASINS lev08 table) truncated to 3 / 4 digits;
areas whose ID is not in the lake-version table fall back to point-in-polygon of their representative point.
Basin of a HeiGIT plant: level-9 HYBAS_ID -> PFAF_ID truncated; fallback point-in-polygon (as for Figure 1a).
Cartogram geometry: each basin polygon is scaled about its centroid so that its area is proportional to P_non;
the reference density D_ref (people per km2 at full size) is the 95th percentile of basin densities, scale <= 1.

Usage: python 05_service.py <data_dir> <hb_dir> <agg_dir> <out_dir> [radius_tag]
"""

import sys, json
import numpy as np, pandas as pd, geopandas as gpd
from shapely import affinity
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig1_config import TAG  # noqa: E402  (config.yaml, section figure1)

DATA, HB, AGG, OUT = sys.argv[1:5]
TAG = sys.argv[5] if len(sys.argv) > 5 else TAG
EA = "EPSG:8857"
DREF_Q = 0.95

ns = pd.read_csv(f"{DATA}/non_served_points.csv", keep_default_na=False, na_values=[""])
hp = pd.read_csv(f"{DATA}/heigit_points.csv", keep_default_na=False, na_values=[""])
pop = pd.read_csv(f"{DATA}/heigit_pop.csv")
hp = hp.merge(pop, on="fid", how="left")
assert hp.zonal_sum_2024.notna().all()
pf8 = pd.read_csv(f"{HB}/hybas_lev08_v1c_attributes.csv")
pf9 = pd.read_csv(f"{HB}/hybas_lev09_v1c_attributes.csv")
pf8 = dict(zip(pf8.HYBAS_ID.astype("int64"), pf8.PFAF_ID.astype("int64")))
pf9 = dict(zip(pf9.HYBAS_ID.astype("int64"), pf9.PFAF_ID.astype("int64")))
cty = gpd.read_file(f"{AGG}/units_country_{TAG}.gpkg", ignore_geometry=True)
unc_cty = set(cty.loc[cty.status == "uncovered", "iso2"])


def spatial(df, polys):
    g = gpd.GeoDataFrame(df[["lon", "lat"]].copy(), geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
    j = gpd.sjoin(g, polys[["PFAF_ID", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    miss = j.PFAF_ID.isna()
    if miss.any():
        nn = gpd.sjoin_nearest(
            g[miss].to_crs(EA), polys[["PFAF_ID", "geometry"]].to_crs(EA), how="left", max_distance=25000
        )
        nn = nn[~nn.index.duplicated()]
        j.loc[miss, "PFAF_ID"] = nn.PFAF_ID.values
    return j.PFAF_ID.values


def pfaf_route(ids, table, lev):
    v = pd.to_numeric(ids, errors="coerce").map(lambda x: table.get(int(x)) if pd.notna(x) else None)
    return v.map(lambda x: int(str(int(x))[:lev]) if pd.notna(x) else np.nan)


report = {
    "non_served_rows": len(ns),
    "non_served_pop": float(ns.pop_sum.sum()),
    "heigit_plants": len(hp),
    "served_pop_2024": float(hp.zonal_sum_2024.sum()),
}
for lev in (3, 4):
    units = gpd.read_file(f"{AGG}/units_hybas{lev}_{TAG}.gpkg")[["PFAF_ID", "status", "geometry"]]
    units = units.rename(columns={"status": "status_1a"})
    polys = gpd.read_file(f"{HB}/hybas_lev{lev:02d}_v1c.gpkg")[["PFAF_ID", "geometry"]]
    polys["PFAF_ID"] = polys.PFAF_ID.astype("int64")
    polys["geometry"] = polys.geometry.make_valid()
    polys = polys.dissolve(by="PFAF_ID", as_index=False)

    # --- non-served areas
    a_id = pfaf_route(ns.HYBAS_ID, pf8, lev)
    a_sp = pd.Series(spatial(ns, polys), index=ns.index)
    both = a_id.notna() & a_sp.notna()
    ns_unit = a_id.where(a_id.notna(), a_sp)
    # --- HeiGIT plants
    h_id = pfaf_route(hp.HYBAS_ID, pf9, lev)
    h_sp = pd.Series(spatial(hp, polys), index=hp.index)
    h_unit = h_id.where(h_id.notna(), h_sp)

    agg = pd.DataFrame(
        {
            "P_non": ns.groupby(ns_unit).pop_sum.sum(),
            "n_areas": ns.groupby(ns_unit).size(),
            "P_served": hp.groupby(h_unit).zonal_sum_2024.sum(),
            "n_plants": hp.groupby(h_unit).size(),
        }
    ).fillna(0)
    agg.index = agg.index.astype("int64")
    u = units.merge(agg, left_on="PFAF_ID", right_index=True, how="left")
    for c in ("P_non", "n_areas", "P_served", "n_plants"):
        u[c] = u[c].fillna(0)
    tot = u.P_served + u.P_non
    u["S"] = np.where(tot > 0, u.P_served / tot.replace(0, np.nan), np.nan)
    u["status"] = np.select([u.status_1a == "uncovered", tot == 0], ["uncovered", "no_data"], "ok")

    # --- cartogram: shrink each basin about its centroid so area ~ P_non
    ue = u.to_crs(EA)
    area_km2 = ue.geometry.area / 1e6
    u["area_km2"] = area_km2.values
    dens = u.P_non / u.area_km2.replace(0, np.nan)
    ok = (u.status == "ok") & (u.P_non > 0)
    dref = float(np.nanquantile(dens[ok], DREF_Q))
    u["shrink"] = np.where(ok, np.minimum(1.0, np.sqrt(dens / dref)), np.nan)
    shrunk = [
        affinity.scale(g, s, s, origin="centroid") if ok_ else None
        for g, s, ok_ in zip(ue.geometry, u["shrink"].fillna(0), ok)
    ]
    sh = gpd.GeoDataFrame(u.drop(columns="geometry"), geometry=gpd.GeoSeries(shrunk, crs=EA), crs=EA)
    u.to_file(f"{OUT}/service_hybas{lev}.gpkg", driver="GPKG")
    sh[ok.values].to_file(f"{OUT}/service_hybas{lev}_cartogram.gpkg", driver="GPKG")
    ok_u = u[u.status == "ok"]
    report[f"lev{lev}"] = dict(
        basins_ok=int((u.status == "ok").sum()),
        uncovered=int((u.status == "uncovered").sum()),
        no_data=int((u.status == "no_data").sum()),
        non_served_by_id=round(float(a_id.notna().mean()), 4),
        non_served_pop_by_id=round(float(ns.pop_sum[a_id.notna()].sum() / ns.pop_sum.sum()), 4),
        id_vs_point_agreement=round(float((a_id[both] == a_sp[both]).mean()), 4),
        non_served_unassigned_pop=float(ns.pop_sum[ns_unit.isna()].sum()),
        heigit_unassigned=int(h_unit.isna().sum()),
        P_non_in_units=float(u.P_non.sum()),
        P_served_in_units=float(u.P_served.sum()),
        P_non_in_uncovered=float(u.loc[u.status == "uncovered", "P_non"].sum()),
        D_ref_people_per_km2=round(dref, 1),
        n_full_size=int((u["shrink"] >= 0.999).sum()),
        S_quantiles={str(q): round(float(ok_u.S.quantile(q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        S_global=round(float(ok_u.P_served.sum() / (ok_u.P_served.sum() + ok_u.P_non.sum())), 4),
        n_S_zero=int((ok_u.S == 0).sum()),
        n_S_one=int((ok_u.S == 1).sum()),
    )
    print(json.dumps(report[f"lev{lev}"], indent=1))
json.dump(report, open(f"{OUT}/service_report.json", "w"), indent=1)
print(json.dumps({k: v for k, v in report.items() if not k.startswith("lev")}, indent=1))
