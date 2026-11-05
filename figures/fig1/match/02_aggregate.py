"""Aggregate reconciled plants to map units (countries now; HydroBASINS L3/L4 once polygons exist).

Per unit:
  n_ref        reference plants (HydroWASTE + UWWTD, deduplicated)
  n_heigit     HeiGIT plants
  n_new        HeiGIT plants without reference counterpart
  n_missing    reference plants without HeiGIT counterpart
  n_corr       ID-linked (corrected) plants;  med_shift / q25 / q75 of their shift [m]
  n_match      spatially matched plants;      med_shift_all = median over corrected+matched
  p_new        n_new / n_heigit       (share of HeiGIT plants that are additions)
  p_missing    n_missing / n_ref      (share of reference plants not found)
  status       'ok' | 'uncovered' (reference plants but no HeiGIT plant) | 'no_ref'

Usage: python 02_aggregate.py <data_dir> <recon_dir> <out_dir> [radius_tag]
"""
import sys
import numpy as np, pandas as pd, geopandas as gpd
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig1_config import MATCH_RADIUS_M, MIN_PLANTS, PATHS, TAG  # noqa: E402  (config.yaml, section figure1)

DATA, REC, OUT = sys.argv[1:4]
TAG = sys.argv[4] if len(sys.argv) > 4 else TAG

# ------------------------------------------------------------------ plants
h = pd.read_csv(f"{REC}/plants_heigit_{TAG}.csv", keep_default_na=False, na_values=[""])
m = pd.read_csv(f"{REC}/plants_missing_{TAG}.csv", keep_default_na=False, na_values=[""])
h["unit_kind"] = "heigit"; m["unit_kind"] = "reference"
pts = pd.concat([h[["lon", "lat", "cls", "shift_m", "HYBAS_ID", "ISO_2"]],
                 m[["lon", "lat", "cls", "ISO_2"]]], ignore_index=True)
# reference plants of every class (for n_ref) are counted from the reference table
ref = pd.read_csv(f"{REC}/reference_plants_{TAG}.csv", keep_default_na=False, na_values=[""])

# ------------------------------------------------------------------ countries (Natural Earth 1:50m)
ne = gpd.read_file(f"{DATA}/ne_50m_admin_0_countries.geojson")
ne["iso2"] = ne["ISO_A2_EH"].where(ne["ISO_A2_EH"] != "-99", ne["ADM0_A3"])
ne = ne[["iso2", "ADMIN", "ADM0_A3", "CONTINENT", "geometry"]].rename(columns={"ADMIN": "name"})
# one polygon per code (AU has three Natural Earth units)
ne = ne.dissolve(by="iso2", aggfunc="first", as_index=False)


def assign(df, polys, key, max_km=25):
    """Point-in-polygon, falling back to the nearest polygon within max_km (coastal outfalls)."""
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
    j = gpd.sjoin(g, polys[[key, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    miss = j[key].isna()
    if miss.any():
        gm = g[miss].to_crs(8857); pm = polys[[key, "geometry"]].to_crs(8857)
        nn = gpd.sjoin_nearest(gm, pm, how="left", max_distance=max_km * 1000)
        nn = nn[~nn.index.duplicated()]
        j.loc[miss, key] = nn[key].values
    return j[key].values


pts["unit"] = assign(pts, ne, "iso2")
ref["unit"] = assign(ref, ne, "iso2")
print("unassigned plants:", int(pd.isna(pts.unit).sum()), "unassigned reference:", int(pd.isna(ref.unit).sum()))



def aggregate(pts, ref, units, key):
    a = pd.DataFrame({
        "n_heigit": pts[pts.cls != "missing"].groupby("unit").size(),
        "n_new": pts[pts.cls == "new"].groupby("unit").size(),
        "n_corr": pts[pts.cls == "corrected"].groupby("unit").size(),
        "n_match": pts[pts.cls == "matched"].groupby("unit").size(),
        "med_shift": pts[pts.cls == "corrected"].groupby("unit").shift_m.median(),
        "q25_shift": pts[pts.cls == "corrected"].groupby("unit").shift_m.quantile(.25),
        "q75_shift": pts[pts.cls == "corrected"].groupby("unit").shift_m.quantile(.75),
        "med_shift_all": pts[pts.cls.isin(["corrected", "matched"])].groupby("unit").shift_m.median(),
    })
    r = pd.DataFrame({"n_ref": ref.groupby("unit").size(),
                      "n_missing": ref[ref.cls == "missing"].groupby("unit").size()})
    a = a.join(r, how="outer")
    cnt = [c for c in a.columns if c.startswith("n_")]
    a[cnt] = a[cnt].fillna(0).astype(int)
    a["p_new"] = a.n_new / a.n_heigit.replace(0, np.nan)
    a["p_missing"] = a.n_missing / a.n_ref.replace(0, np.nan)
    a["status"] = np.select([a.n_ref == 0, a.n_heigit == 0], ["no_ref", "uncovered"], "ok")
    u = units.merge(a, left_on=key, right_index=True, how="left")
    u["status"] = u["status"].fillna("no_ref")
    u[cnt] = u[cnt].fillna(0).astype(int)
    return u


cty = aggregate(pts, ref, ne, "iso2")
cty.to_file(f"{OUT}/units_country_{TAG}.gpkg", driver="GPKG")
pts.to_csv(f"{OUT}/plants_with_units_{TAG}.csv", index=False)
ref.to_csv(f"{OUT}/reference_with_units_{TAG}.csv", index=False)

# ------------------------------------------------------------------ uncovered-country reference plants -> GeoJSON
unc = set(cty.loc[cty.status == "uncovered", "iso2"])
W = pd.read_csv(f"{DATA}/HydroWASTE_v10.csv", encoding="latin-1", keep_default_na=False, na_values=[""])
U = pd.read_csv(f"{DATA}/uwwtd_points.csv", keep_default_na=False, na_values=[""])
# countries added on request although HeiGIT has plants there
EXTRA = {"ZA": "HeiGIT kept the HydroWASTE coordinates (no location correction)"}
# one feature per source record (its own ID and coordinates); class and country are those of its reference plant
recs = pd.read_csv(f"{REC}/reference_records_{TAG}.csv", keep_default_na=False, na_values=[""])
sel = ref[ref.unit.isin(unc | set(EXTRA))].set_index("ref_id")
recs = recs[recs.ref_id.isin(sel.index)]
has_hw = set(recs.loc[recs.src == "HW", "ref_id"])
recs = recs[(recs.src == "HW") | ~recs.ref_id.isin(has_hw)]       # UWWTD record only where the plant has no HydroWASTE record
uw_rep = {r.Index: r.uwwCode for r in sel.itertuples() if isinstance(r.uwwCode, str) and r.uwwCode}
uw_name = dict(zip(U.uwwCode, U.uwwName))
rows = []
for r in recs.itertuples():
    p = sel.loc[r.ref_id]
    d = dict(country_iso2=p.unit, reason=EXTRA.get(p.unit, "no HeiGIT coverage"), ref_class=p.cls,
             ref_src=p.ref_src, lon=r.lon, lat=r.lat)
    if r.src == "HW":
        w = W.iloc[int(r.row)]
        d.update(HW_WASTE_ID=int(w.WASTE_ID), WWTP_NAME=w.WWTP_NAME, COUNTRY=w.COUNTRY, CNTRY_ISO=w.CNTRY_ISO,
                 STATUS=w.STATUS, POP_SERVED=w.POP_SERVED, WASTE_DIS=w.WASTE_DIS, LEVEL=w.LEVEL, QUAL_LOC=w.QUAL_LOC)
        if r.ref_id in uw_rep:
            d.update(uwwCode=uw_rep[r.ref_id], uwwName=uw_name.get(uw_rep[r.ref_id]))
    else:
        d.update(uwwCode=r.uwwCode, uwwName=uw_name.get(r.uwwCode))
    rows.append(d)
gj = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=gpd.points_from_xy([r["lon"] for r in rows], [r["lat"] for r in rows]), crs=4326)
gj.to_file(f"{OUT}/reference_plants_without_heigit_coverage.geojson", driver="GeoJSON")

ok = cty[cty.status == "ok"]
print("countries ok:", len(ok), "uncovered:", sorted(unc), "features in GeoJSON:", len(gj), gj.reason.value_counts().to_dict(),
      "plants:", len(sel), sel.unit.isin(set(EXTRA)).value_counts().to_dict())
print("p_new quantiles", ok.p_new.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
print("p_missing quantiles", ok.p_missing.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
print("med_shift quantiles", ok.med_shift.quantile([.1, .25, .5, .75, .9]).round(0).to_dict())
print("n_ref<5:", int((ok.n_ref < 5).sum()), " n_corr<5:", int((ok.n_corr < 5).sum()), " n_corr==0:", int((ok.n_corr == 0).sum()))
print("totals: ref", ok.n_ref.sum() + cty.loc[cty.status == 'uncovered', 'n_ref'].sum(), "heigit", cty.n_heigit.sum(),
      "new", cty.n_new.sum(), "missing", cty.n_missing.sum(), "corr", cty.n_corr.sum(), "match", cty.n_match.sum())
print(ok.sort_values("n_ref", ascending=False)[["iso2", "n_ref", "n_heigit", "n_new", "n_missing", "n_corr", "n_match", "p_new", "p_missing", "med_shift"]].head(25).round(2).to_string())
