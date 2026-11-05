"""Repair HeiGIT plant coordinates that were back-projected from UTM with the wrong zone/hemisphere.

Symptom: an ID-linked HeiGIT plant lies in another UTM zone than its HydroWASTE/UWWTD record, at the same
latitude, with the longitude off by a whole number of zones (6 deg each), or in the wrong hemisphere.
Test for each ID-linked plant (reference record chosen as in 01_reconcile: UWWTD for UWW-sourced plants):
  1. the HeiGIT point and the reference record are in different UTM zones and more than 50 km apart;
  2. projecting the HeiGIT point to UTM with the zone it lies in, and back with the reference's zone,
     brings it within 5 km of the reference record.
Plants passing both get the re-projected coordinate (this keeps HeiGIT's own small correction), their level-9
HYBAS_ID is cleared so their basin is taken from the repaired location, and they are flagged utm_fixed = 1.

Usage: python 00_fix_utm.py <data_dir>
Reads  heigit_points_original.csv (created from heigit_points.csv on first run), writes heigit_points.csv
and utm_fixes.csv (one row per repaired plant).
"""
import os, sys
import numpy as np, pandas as pd, geopandas as gpd, pycountry
from pyproj import Transformer

D = sys.argv[1]
orig = f"{D}/heigit_points_original.csv"
if not os.path.exists(orig):
    os.rename(f"{D}/heigit_points.csv", orig)
H = pd.read_csv(orig, dtype={"uwwCode": str, "ISO_2": str}, keep_default_na=False, na_values=[""])
W = pd.read_csv(f"{D}/HydroWASTE_v10.csv", encoding="latin-1", keep_default_na=False, na_values=[""]).set_index("WASTE_ID")
U = pd.read_csv(f"{D}/uwwtd_points.csv", keep_default_na=False, na_values=[""]).set_index("uwwCode")
E = pd.read_csv(f"{D}/heigit_epsg.csv")


def zone_epsg(lon, lat):
    z = np.clip(np.floor((np.asarray(lon) + 180) / 6).astype(int) + 1, 1, 60)
    return np.where(np.asarray(lat) >= 0, 32600 + z, 32700 + z)


def hav(lo1, la1, lo2, la2):
    lo1, la1, lo2, la2 = map(np.radians, (lo1, la1, lo2, la2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


hasU = H.uwwCode.notna() & H.uwwCode.isin(U.index)
hasW = H.HW_WASTE_ID.notna()
useU = hasU & ((H.SOURCE_DS == "UWW") | ~hasW)
wi = H.HW_WASTE_ID.astype("Int64")
H["ref_lon"] = np.where(useU, U.reindex(H.uwwCode).lon.values, W.reindex(wi).LON_WWTP.values)
H["ref_lat"] = np.where(useU, U.reindex(H.uwwCode).lat.values, W.reindex(wi).LAT_WWTP.values)
a2 = lambda x: (pycountry.countries.get(alpha_3=x).alpha_2 if isinstance(x, str) and pycountry.countries.get(alpha_3=x)
                else {"XKX": "XK"}.get(x))
H["ref_country"] = np.where(useU, U.reindex(H.uwwCode).ISO_2_UWW.replace({"EL": "GR"}).values,
                            W.reindex(wi).CNTRY_ISO.map(a2).values)
L = H[hasU | hasW].copy()
L["shift_m"] = hav(L.lon, L.lat, L.ref_lon, L.ref_lat)
L["zone_heigit"] = zone_epsg(L.lon, L.lat)
L["zone_ref"] = zone_epsg(L.ref_lon, L.ref_lat)
C = L[(L.zone_heigit != L.zone_ref) & (L.shift_m > 50_000)].copy()
fx, fy = [], []
for _, r in C.iterrows():
    e, n = Transformer.from_crs(4326, int(r.zone_heigit), always_xy=True).transform(r.lon, r.lat)
    lo, la = Transformer.from_crs(int(r.zone_ref), 4326, always_xy=True).transform(e, n)
    fx.append(lo); fy.append(la)
C["fixed_lon"], C["fixed_lat"] = fx, fy
C["fixed_shift_m"] = hav(C.fixed_lon, C.fixed_lat, C.ref_lon, C.ref_lat)
fix = C[C.fixed_shift_m < 5000].copy()

# country at the old and at the repaired location (Natural Earth 1:50m, nearest within 30 km)
ne = gpd.read_file(f"{D}/ne_50m_admin_0_countries.geojson")
ne["iso2"] = ne.ISO_A2_EH.where(ne.ISO_A2_EH != "-99", ne.ADM0_A3)
ne = ne[["iso2", "geometry"]].to_crs(8857)
def country(lon, lat):
    g = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326).to_crs(8857)
    j = gpd.sjoin_nearest(g, ne, how="left", max_distance=30000)
    return j[~j.index.duplicated()].iso2.values
fix["country_old"] = country(fix.lon, fix.lat)
fix["country_new"] = country(fix.fixed_lon, fix.fixed_lat)
fix = fix.merge(E, on="fid", how="left")

out = H.drop(columns=["ref_lon", "ref_lat", "ref_country"]).copy()
out["utm_fixed"] = 0
idx = out.fid.isin(fix.fid)
m = fix.set_index("fid")
out.loc[idx, "lon"] = out.loc[idx, "fid"].map(m.fixed_lon).round(6)
out.loc[idx, "lat"] = out.loc[idx, "fid"].map(m.fixed_lat).round(6)
out.loc[idx, "ISO_2"] = out.loc[idx, "fid"].map(m.country_new)
out.loc[idx, "HYBAS_ID"] = np.nan
out.loc[idx, "utm_fixed"] = 1
out.to_csv(f"{D}/heigit_points.csv", index=False)
cols = ["fid", "SOURCE_DS", "HW_WASTE_ID", "uwwCode", "ref_country", "country_old", "country_new", "epsg",
        "zone_heigit", "zone_ref", "lat", "lon", "fixed_lat", "fixed_lon", "ref_lat", "ref_lon", "shift_m", "fixed_shift_m"]
fix[cols].to_csv(f"{D}/utm_fixes.csv", index=False)
print(f"candidates (other zone, > 50 km): {len(C)}; repaired (within 5 km after re-projection): {len(fix)}")
print(f"repaired shift: max {fix.fixed_shift_m.max():.0f} m, median {fix.fixed_shift_m.median():.0f} m;"
      f" country now matches reference: {int((fix.country_new == fix.ref_country).sum())}/{len(fix)}")
