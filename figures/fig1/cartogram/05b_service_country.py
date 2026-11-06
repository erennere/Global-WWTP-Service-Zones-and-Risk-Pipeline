"""Served share of the urban population per Natural Earth country (map 2 at country level, Figure 1c/d).

  P_served = sum of HeiGIT service-area population (2024_zonal_sum) of the plants in the country
  P_non    = sum of pop_sum of the non-served urban areas in the country
  S        = P_served / (P_served + P_non),  R_not_served = 1 - S

Countries are Natural Earth 1:50m units, as in map 1 (one polygon per ISO code). Plants and urban areas are
assigned by point-in-polygon, with the nearest country within 25 km as fallback (as 02_aggregate.py).
status: uncovered (map 1 status) / no_data (no urban population) / ok.

Usage: python 05b_service_country.py <data_dir> <agg_dir> <out_dir> [radius_tag]
  <data_dir>  non_served_points.csv, heigit_points.csv, heigit_pop.csv, ne_50m_admin_0_countries.geojson
  <agg_dir>   map 1 aggregation (units_country_<tag>.gpkg, for the coverage status)
Writes <out_dir>/service_country.gpkg and service_country_table.csv, and adds a "country" entry to
service_report.json.
"""

import sys, json, os
import numpy as np, pandas as pd, geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig1_config import TAG  # noqa: E402  (config.yaml, section figure1)

DATA, AGG, OUT = sys.argv[1:4]
TAG = sys.argv[4] if len(sys.argv) > 4 else TAG
EA = "EPSG:8857"

ns = pd.read_csv(f"{DATA}/non_served_points.csv", keep_default_na=False, na_values=[""])
hp = pd.read_csv(f"{DATA}/heigit_points.csv", keep_default_na=False, na_values=[""])
hp = hp.merge(pd.read_csv(f"{DATA}/heigit_pop.csv"), on="fid", how="left")
assert hp.zonal_sum_2024.notna().all()

ne = gpd.read_file(f"{DATA}/ne_50m_admin_0_countries.geojson")
ne["iso2"] = ne["ISO_A2_EH"].where(ne["ISO_A2_EH"] != "-99", ne["ADM0_A3"])
ne = ne[["iso2", "ADMIN", "ADM0_A3", "ISO_A3_EH", "CONTINENT", "geometry"]].rename(columns={"ADMIN": "name"})
ne = ne.dissolve(by="iso2", aggfunc="first", as_index=False)
cty = gpd.read_file(f"{AGG}/units_country_{TAG}.gpkg", ignore_geometry=True)
status_1a = dict(zip(cty.iso2, cty.status))


def assign(df, polys, key, max_km=25):
    """Point-in-polygon, falling back to the nearest polygon within max_km (coastal points)."""
    g = gpd.GeoDataFrame(df[["lon", "lat"]].copy(), geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
    j = gpd.sjoin(g, polys[[key, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    miss = j[key].isna()
    if miss.any():
        nn = gpd.sjoin_nearest(
            g[miss].to_crs(EA), polys[[key, "geometry"]].to_crs(EA), how="left", max_distance=max_km * 1000
        )
        nn = nn[~nn.index.duplicated()]
        j.loc[miss, key] = nn[key].values
    return j[key].values


ns["unit"] = assign(ns, ne, "iso2")
hp["unit"] = assign(hp, ne, "iso2")

# agreement with the non-served file's own country column (whichever code it uses)
agree = None
if "country" in ns.columns:
    got = ns.unit.notna() & ns.country.notna()
    best = 0.0
    for col in ("iso2", "ADM0_A3", "ISO_A3_EH", "name"):
        m = dict(zip(ne.iso2, ne[col].astype(str)))
        best = max(
            best,
            float((ns.loc[got, "unit"].map(m).str.upper() == ns.loc[got, "country"].astype(str).str.upper()).mean()),
        )
    agree = round(best, 4)

agg = pd.DataFrame(
    {
        "P_non": ns.groupby("unit").pop_sum.sum(),
        "n_areas": ns.groupby("unit").size(),
        "P_served": hp.groupby("unit").zonal_sum_2024.sum(),
        "n_plants": hp.groupby("unit").size(),
    }
).fillna(0)
u = ne.merge(agg, left_on="iso2", right_index=True, how="left")
for c in ("P_non", "n_areas", "P_served", "n_plants"):
    u[c] = u[c].fillna(0)
for c in ("n_areas", "n_plants"):
    u[c] = u[c].astype(int)
tot = u.P_served + u.P_non
u["S"] = np.where(tot > 0, u.P_served / tot.replace(0, np.nan), np.nan)
u["R_not_served"] = 1 - u.S
u["status_1a"] = u.iso2.map(status_1a).fillna("no_ref")
u["status"] = np.select([u.status_1a == "uncovered", tot == 0], ["uncovered", "no_data"], "ok")
u.to_file(f"{OUT}/service_country.gpkg", driver="GPKG")
u.drop(columns="geometry").to_csv(f"{OUT}/service_country_table.csv", index=False)

ok = u[u.status == "ok"]
rep = dict(
    countries_ok=int(len(ok)),
    uncovered=int((u.status == "uncovered").sum()),
    no_data=int((u.status == "no_data").sum()),
    non_served_unassigned_pop=float(ns.pop_sum[ns.unit.isna()].sum()),
    heigit_unassigned=int(hp.unit.isna().sum()),
    country_column_agreement=agree,
    P_non_in_units=float(u.P_non.sum()),
    P_served_in_units=float(u.P_served.sum()),
    P_non_in_uncovered=float(u.loc[u.status == "uncovered", "P_non"].sum()),
    R_global=round(float(ok.P_non.sum() / (ok.P_served.sum() + ok.P_non.sum())), 4),
    top_non_served=ok.sort_values("P_non", ascending=False)
    .head(10)[["iso2", "name", "P_non"]]
    .round(0)
    .to_dict("records"),
)
path = f"{OUT}/service_report.json"
report = json.load(open(path)) if os.path.exists(path) else {}
report["country"] = rep
json.dump(report, open(path, "w"), indent=1)
print(json.dumps(rep, indent=1))
